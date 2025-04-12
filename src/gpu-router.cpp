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
#include <arpa/inet.h>

constexpr size_t BURST_SIZE = 32;
constexpr size_t PACKET_SIZE = 1519;

using Packet = std::array<uint8_t, PACKET_SIZE>;
using PacketBatch = std::vector<Packet>;
using ArrayCounter = std::array<u_int8_t, BURST_SIZE>;
using InspectResult = std::pair<PacketBatch, std::array<ArrayCounter, 6>>;

/**
 * Parse the packet and determine its type
 * returns protocol index
 */
int parse_packet(const unsigned char* packet) {
    struct ether_header* eth_hdr = (struct ether_header*) packet;
    struct iphdr* ip_hdr = (struct iphdr*) (packet + sizeof(struct ether_header));
    switch (ntohs(eth_hdr->ether_type)) {
        case ETHERTYPE_IP:
            switch (ip_hdr->protocol) {
                case IPPROTO_ICMP:
                    return ICMP_IDX;
                case IPPROTO_TCP:
                    return TCP_IDX;
                case IPPROTO_UDP:
                    return UDP_IDX;
                default:
                    return -1;
            }
        case ETHERTYPE_ARP:
            return ARP_IDX;
        case ETHERTYPE_IPV6:
            return IPv6_IDX;
        default:
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
    pcap_t *handle{};

    if (argc == 1) {
        handle = pcap_open_offline("/root/keysight-challenge2025/src/capture2.pcap", errbuf);
    } else {
        // handle = pcap_open_live(argv[1], , , ,errbuf);
    }
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

    // Packet inspection node
    tbb::flow::function_node<PacketBatch, PacketBatch> inspect_packet_node {
        g, tbb::flow::unlimited, [&](PacketBatch batch) -> InspectResult {
            PacketBatch result(batch.size());
            std::array<std::array<u_int8_t, BURST_SIZE>, 6> counters{std::array<u_int8_t, BURST_SIZE>{}};

            // By including all the SYCL work in a {} block, we ensure
            // all SYCL tasks must complete before exiting the block
            {
                // Create a SYCL buffer from the counters array
                sycl::buffer<u_int8_t, 2> counters_buf(reinterpret_cast<u_int8_t*>(counters.data()),
                                                       sycl::range<2>(6, BURST_SIZE));

                                                       sycl::queue gpuQ(sycl::default_selector_v, dpc_common::exception_handler);
                
                sycl::range<1> n_items{batch.size()}; 
                sycl::buffer batch_buffer(batch);
                sycl::buffer result_buffer(result);

                gpuQ.submit([&](sycl::handler& h) {
                    sycl::accessor counters_acc(counters_buf, h, sycl::read_write);
                    sycl::accessor batch_acc(batch_buffer, h, sycl::read_only);
                    sycl::accessor result_acc(result_buffer, h, sycl::write_only);

                    sycl::stream out(1024, 256, h); // buffer size, message size
                    auto compute = [=](auto i) {
                        // Process the packets
                        const unsigned char* packet = batch_acc[i].data();
                        int8_t packet_type = parse_packet(packet);
                        // out << "Packet " << i[0] << " type: " << packet_type << ".\n";
                        if (packet_type != -1) {
                            if (packet_type == ICMP_IDX || 
                                packet_type == TCP_IDX ||
                                packet_type == UDP_IDX) {
                                counters_acc[IPv4_IDX][i]++;
                            }
                            result_acc[i] = batch_acc[i];
                            counters_acc[packet_type][i]++;
                        }
                    };
                    h.parallel_for(n_items, compute);
                }).wait_and_throw();  // end of the commands for the SYCL queue
                std::cout << "SYCL block done\n";
                for (int i = 0; i < 6; ++i) {
                    std::cout << "Counter " << i << ": ";
                    for (int j = 0; j < BURST_SIZE; ++j) {
                        std::cout << static_cast<int>(counters[i][j]) << " ";
                    }
                    std::cout << std::endl;
                }
            }  // End of the scope for SYCL code; the queue has completed the work
            // Return the number of packets processed
            
            return {result, counters};
        }
    };

    tbb::flow::function_node<PacketBatch, PacketBatch> routing_node{
        g, tbb::flow::unlimited,
        [&](const InspectResult& inspect_result) -> PacketBatch {
            auto batch = inspect_result.first;
            {
                sycl::queue gpuQ(sycl::default_selector_v, dpc_common::exception_handler);

                sycl::range<1> n_items{batch.size()}; 
                sycl::buffer batch_buffer(batch);
                gpuQ.submit([&](sycl::handler& h) {
                    sycl::accessor batch_accessor(batch_buffer, h, sycl::read_write);
                    auto compute = [=](sycl::id<1> index) {
                        auto packet = batch_accessor[index];
                        ether_header* eth = (ether_header*)packet.data();
                        if (eth->ether_type == ETHERTYPE_IP) {
                            return;
                        }
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
    tbb::flow::function_node<int, std::array<int, 6>> sum_counters_node {
        g, tbb::flow::unlimited, [&](const InspectResult &result) {
            std::array<int, 6> total_counters = {0, 0, 0, 0, 0, 0};
            
            // TODO: Parallel reduce
            for (const auto &c : total_counters) {
                std::cout << c << ' ';
            }
            std::cout << '\n';
        }
    };
    
    // construct graph
    tbb::flow::make_edge<PacketBatch>(in_node, inspect_packet_node);
    tbb::flow::make_edge<PacketBatch>(inspect_packet_node, routing_node);
    tbb::flow::make_edge<PacketBatch>(inspect_packet_node, sum_counters_node);
    tbb::flow::make_edge<PacketBatch>(routing_node, send_node);

    in_node.activate();
    g.wait_for_all();

    std::cout << "Done waiting" << std::endl;
}
