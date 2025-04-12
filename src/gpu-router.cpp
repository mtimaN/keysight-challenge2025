#include <array>
#include <vector>
#include <iostream>
#include <string>

#include <sycl/sycl.hpp>

#include <pcap.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/flow_graph.h>
#include "dpc_common.hpp"
#include "protocols.h"

constexpr size_t BURST_SIZE = 32;
constexpr size_t PACKET_SIZE = 1519;

using Packet = std::array<uint8_t, PACKET_SIZE>;
using PacketBatch = std::vector<Packet>;

/**
 * Parse the packet and determine its type
 * returns protocol index
 */
int parse_packet(const char* packet, int idx) {
    struct ether_header* eth_hdr = (struct ether_header*) packet;

    switch (ntohs(eth_hdr->ether_type)) {
        case ETHERTYPE_IP:
            struct iphdr* ip_hdr = (struct iphdr*) (packet + sizeof(struct ether_header));
            std::cout << "IP packet" << std::endl;
            switch (ntoh(ip_hdr->protocol)) {
                case IPPROTO_ICMP:
                    return ICMP_IDX;
                case IPPROTO_TCP:
                    std::cout << "TCP packet" << std::endl;
                    return TCP_IDX;
                case IPPROTO_UDP:
                    std::cout << "UDP packet" << std::endl;
                    return UDP_IDX;
                default:
                    std::cout << "Unknown protocol" << std::endl;
                    return -1;
            }
        case ETHERTYPE_ARP:
            std::cout << "ARP packet" << std::endl;
            return ARP_IDX;
        case ETHERTYPE_IPV6:
            std::cout << "IPv6 packet" << std::endl;
            return IPv6_IDX;
        default:
            std::cout << "Unknown Ethernet type" << std::endl;
            return -1;
    }
}

int main(int argc, char* argv[]) {
    sycl::queue q;

    std::cout << "Using device: " <<
        q.get_device().get_info<sycl::info::device::name>() << std::endl;

    char errbuf[PCAP_ERRBUF_SIZE];
    int nth = 10;  // number of threads
    auto mp = tbb::global_control::max_allowed_parallelism;
    pcap_t* handle = pcap_open_offline("/root/keysight-challenge2025/src/capture2.pcap", errbuf);
    tbb::global_control gc(mp, nth);
    tbb::flow::graph g;

    // Input node: get packets from the socket or from the packet capture
    tbb::flow::input_node<PacketBatch> in_node{g,
        [&](tbb::flow_control& fc) -> PacketBatch {
            PacketBatch batch;

            for (size_t i = 0; i < BURST_SIZE; ++i) {
                struct pcap_pkthdr* header;
                const u_char* pkt;
                int ret = pcap_next_ex(handle, &header, &pkt);
                if (ret == 1 && header && pkt) {
                    Packet p{};
                    size_t len = PACKET_SIZE;
                    std::memcpy(p.data(), pkt, len);
                    batch.push_back(p);
                } else if (ret == -2 || ret == -1) {
                    fc.stop();
                    break;
                }
            }

            return batch;
        }
    };

    std::array<std::array<u_int8_t, burst_size>, 6> counters;
    // Packet inspection node
    tbb::flow::function_node<PacketBatch> inspect_packet_node {
        g, tbb::flow::unlimited, [&](PacketBatch batch) -> PacketBatch {
            PacketBatch result;

            // By including all the SYCL work in a {} block, we ensure
            // all SYCL tasks must complete before exiting the block
            {
                // Create a SYCL buffer from the counters array
                sycl::buffer<u_int8_t, 2> counters_buf(reinterpret_cast<u_int8_t*>(counters.data()),
                                                       sycl::range<2>(6, burst_size));

                                                       sycl::queue gpuQ(sycl::gpu_selector_v, dpc_common::exception_handler);

                std::cout << "Selected GPU Device Name: " <<
                gpuQ.get_device().get_info<sycl::info::device::name>() << "\n";
                
                sycl::range<1> n_items{batch.size()}; 
                sycl::buffer batch_buffer(batch);
                sycl::buffer result_buffer(result);

                gpuQ.submit([&](sycl::handler& h) {
                    auto counters_acc = counters_buf.get_access<sycl::access::mode::read_write>(h);
                    auto batch_acc = batch_buffer.get_access<sycl::access::mode::read>(h);
                    auto result_acc = result_buffer.get_access<sycl::access::mode::write>(h);
                    
                    auto compute = [=](auto i) {
                        // Process the packets
                        char* packet = batch_acc[i];
                        uint8_t packet_type = parse_packet(packet);
                        if (packet_type != -1) {
                            if (packet_type == ICMP_IDX || 
                                packet_type == TCP_IDX ||
                                packet_type == UDP_IDX) {
                                counters_acc[IPv4_IDX][i]++;
                                result_acc[i] = packet;
                            }
                            counters_acc[packet_type][i]++;
                        }
                    };

                    h.parallel_for(nr_packets, compute);
                }).wait_and_throw();  // end of the commands for the SYCL queue
            }  // End of the scope for SYCL code; the queue has completed the work
            // Return the number of packets processed
            return result;
        }
    };

    tbb::flow::function_node<PacketBatch, PacketBatch> routing_node{
        g, tbb::flow::unlimited,
        [&](const PacketBatch& batch) -> PacketBatch {
            {
                sycl::queue gpuQ(sycl::default_selector_v, dpc_common::exception_handler);
                std::cout << "Selected Device Name: " <<
                gpuQ.get_device().get_info<sycl::info::device::name>() << "\n";

                sycl::range<1> n_items{batch.size()}; 
                sycl::buffer batch_buffer(batch);
                gpuQ.submit([&](sycl::handler& h) {
                    sycl::accessor batch_accessor(batch_buffer, h, sycl::read_write);
                    auto compute = [=](sycl::id<1> index) {
                        auto packet = batch_accessor[index];
                        ether_header* eth = (ether_header*)packet.data();
                        iphdr* ip = (iphdr*)(packet.data() + sizeof(ether_header));

                        ip->daddr += 1 + (1 << 8) + (1 << 16) + (1 << 24); // Increment destination IP
                    };

                   
                    h.parallel_for(n_items, compute);
                }).wait_and_throw();
            }
            return batch;
        }
    };

    tbb::flow::function_node<PacketBatch, PacketBatch> send_node{
        g, tbb::flow::unlimited,
        [&](const PacketBatch& batch) -> PacketBatch {
            for (const auto& packet : batch) {
                pcap_sendpacket(handle, packet.data(), packet.size());
            }
            return batch;
        }
    };

    // construct graph
    // Sum counters node
    // tbb::flow::function_node<int, std::array<int, 6>> sum_counters_node {
    //     g, tbb::flow::unlimited, [&](int) {
    //         std::array<int, 6> total_counters = {0, 0, 0, 0, 0, 0};

            
            
    //         return total_counters;
    //     }
    // };
    
    // construct graph
    tbb::flow::make_edge<PacketBatch>(in_node, inspect_packet_node);
    tbb::flow::make_edge<PacketBatch>(inspect_packet_node, routing_node);
    tbb::flow::make_edge<PacketBatch>(routing_node, send_node);

    in_node.activate();
    g.wait_for_all();

    std::cout << "Done waiting" << std::endl;
    // Print counters array
    for (int i = 0; i < 6; ++i) {
        std::cout << "Counter " << i << ": ";
        for (int j = 0; j < burst_size; ++j) {
            std::cout << static_cast<int>(counters[i][j]) << " ";
        }
        std::cout << std::endl;
    }
}
