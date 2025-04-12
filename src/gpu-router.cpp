#include <array>
#include <vector>
#include <iostream>
#include <string>

#include <sycl/sycl.hpp>

#include <pcap.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/flow_graph.h>
#include <tbb/parallel_reduce.h>

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
        handle = pcap_open_offline("/root/keysight-challenge2025/src/capture3.pcap", errbuf);
    } else {
        handle = pcap_open_live(argv[1], BUFSIZ, 1, 1000, errbuf);
    }
    tbb::global_control gc(mp, nth);
    tbb::flow::graph g;

    // Input node: get packets from the socket or from the packet capture
    tbb::flow::input_node<PacketBatch> in_node{g,
        [&](tbb::flow_control& fc) -> PacketBatch {
            PacketBatch batch;
            struct pcap_pkthdr* header;
            const u_char* pkt;
            int ret = pcap_next_ex(handle, &header, &pkt);
            if (ret == -2 || ret == -1) {
                fc.stop();
            }
        
            for (size_t i = 0; i < BURST_SIZE; ++i) {
                if (ret == 1 && header && pkt) {
                    Packet p{};
                    size_t len = PACKET_SIZE;
                    std::memcpy(p.data(), pkt, len);
                    batch.push_back(p);
                } else if (ret == -2 || ret == -1) {
                    break;
                }
                ret = pcap_next_ex(handle, &header, &pkt);
            }

            return batch;
        }
    };


    double avg_inspect_time = 0.0;
    int inspect_count = 0;
    // Packet inspection node
    tbb::flow::function_node<PacketBatch, InspectResult> inspect_packet_node {
        g, tbb::flow::unlimited, [&](PacketBatch batch) -> InspectResult {
            PacketBatch result(batch.size());
            std::array<std::array<u_int8_t, BURST_SIZE>, 6> counters{};

            // By including all the SYCL work in a {} block, we ensure
            // all SYCL tasks must complete before exiting the block
            {
                // Create a SYCL buffer from the counters array
                sycl::buffer<u_int8_t, 2> counters_buf(reinterpret_cast<u_int8_t*>(counters.data()),
                                                       sycl::range<2>(6, BURST_SIZE));

                sycl::queue gpuQ(sycl::default_selector_v,
                    sycl::property::queue::enable_profiling());
                
                sycl::range<1> n_items{batch.size()}; 
                sycl::buffer batch_buffer(batch);
                sycl::buffer result_buffer(result);

                sycl::event event = gpuQ.submit([&](sycl::handler& h) {
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
                });
                event.wait();
                double kernel_time = (event.template get_profiling_info<
                    sycl::info::event_profiling::command_end>() -
                event.template get_profiling_info<
                    sycl::info::event_profiling::command_start>()) / 1e6;
                avg_inspect_time += kernel_time;
                inspect_count++;

            }  // End of the scope for SYCL code; the queue has completed the work
            
            return {result, counters};
        }
    };


    double avg_routing_time = 0.0;
    int routing_count = 0;
    tbb::flow::function_node<InspectResult, PacketBatch> routing_node{
        g, tbb::flow::unlimited,
        [&](const InspectResult& inspect_result) -> PacketBatch {
            auto batch = inspect_result.first;
            {
                sycl::queue gpuQ(sycl::default_selector_v,
                    sycl::property::queue::enable_profiling());

                sycl::range<1> n_items{batch.size()}; 
                sycl::buffer batch_buffer(batch);
                sycl::event event = gpuQ.submit([&](sycl::handler& h) {
                    sycl::accessor batch_accessor(batch_buffer, h, sycl::read_write);
                    auto compute = [=](sycl::id<1> index) {
                        auto& packet = batch_accessor[index];
                        ether_header* eth = (ether_header*)packet.data();
                        if (ntohs(eth->ether_type) != ETHERTYPE_IP) {
                            return;
                        }
                        iphdr* ip = (iphdr*)(packet.data() + sizeof(ether_header));

                        ip->daddr += 1 + (1 << 8) + (1 << 16) + (1 << 24); // Increment destination IP
                    };
                   
                    h.parallel_for(n_items, compute);
                });
                
                event.wait();
                double kernel_time = (event.template get_profiling_info<
                    sycl::info::event_profiling::command_end>() -
                event.template get_profiling_info<
                    sycl::info::event_profiling::command_start>()) / 1e6;
                avg_routing_time += kernel_time;
                routing_count++;
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

    // Sum counters node
    std::array<int, 6> total_counters = {0, 0, 0, 0, 0, 0};
    tbb::flow::function_node<InspectResult> sum_counters_node {
        g, tbb::flow::unlimited, [&](const InspectResult &result) {
            auto counters = result.second;

            // TODO: Parallel reduce
            for (int i = 0; i < 6; ++i) {
                total_counters[i] += tbb::parallel_reduce(
                    tbb::blocked_range<size_t>(0, BURST_SIZE),
                    0,
                    [&](const tbb::blocked_range<size_t>& r, int sum) {
                        for (size_t j = r.begin(); j != r.end(); ++j) {
                            sum += counters[i][j];
                        }
                        return sum;
                    },
                    std::plus<int>());
            }
        }
    };
    
    // construct graph - fix the edge connections with correct types
    tbb::flow::make_edge(in_node, inspect_packet_node);
    tbb::flow::make_edge(inspect_packet_node, routing_node);
    tbb::flow::make_edge(inspect_packet_node, sum_counters_node);
    tbb::flow::make_edge(routing_node, send_node);

    in_node.activate();
    g.wait_for_all();

    std::cout << "Done waiting" << std::endl;
    
    // Clean up
    pcap_close(handle);
    for (int i = 0; i < 6; ++i) {
        if (i == 0) {
            std::cout << "IPv4 count: " << total_counters[i] << "\n";
        } else if (i == 1) {
            std::cout << "IPv6 count: " << total_counters[i] << "\n";
        } else if (i == 2) {
            std::cout << "ARP count: " << total_counters[i] << "\n";
        } else if (i == 3) {
            std::cout << "ICMP count: " << total_counters[i] << "\n";
        } else if (i == 4) {
            std::cout << "TCP count: " << total_counters[i] << "\n";
        } else if (i == 5) {
            std::cout << "UDP count: " << total_counters[i] << "\n";
        }
    }
    
    std::cout << "------------------------------------------\n";
    std::cout << "Time profiling:\n\n";
    std::cout << "Average inspection time: " << avg_inspect_time / inspect_count << " ms\n";
    std::cout << "Average routing time: " << avg_routing_time / routing_count << " ms\n";

    return 0;
}