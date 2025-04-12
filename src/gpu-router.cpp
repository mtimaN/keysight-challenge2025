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

int main() {
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
    tbb::flow::make_edge<PacketBatch>(in_node, routing_node);
    tbb::flow::make_edge<PacketBatch>(routing_node, send_node);

    in_node.activate();
    g.wait_for_all();

    std::cout << "Done waiting" << std::endl;
}
