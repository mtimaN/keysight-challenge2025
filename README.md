# Keysight Challenge 2025

## Intro
Welcome to the Keysgiht Challenge 2025. In this challenge you will have to run code on the GPU and demonstrate your skills in parallelizing the code for better performance.

The main description of the task is in this [document](https://docs.google.com/document/d/1-A59iiqdzbKEcdTZGfll-y3Vl6Kw7nMEBiraD2W86pU/edit?usp=sharing).

### On a Linux System
    * Build the gpu-router application
      git clone $YOUR_GITHUB_FORK
      cd keysight-challenge-2025
      mkdir build
      cd build
      cmake ..
      make VERBOSE=1

    * Run the program
      make run

    * Clean the program
      make clean

## Implementation

We have implemented all the main features presented in the document.

Packets are read using libpcap, parsed using a scheduler queue on the GPU, sent to both the IPv4 Router and the Statistics counter(IPv6 packets are discarded by the Router.)

Bonus implementations:
- use of std::array
- use of SYCL buffers and accessors
- use of parallel_reduce()
- time profiling

## Possible improvements

One possible improvement would have been leveraging asynchronous programming while solving the task. The gpu tasks are handled asynchronously, but the CPU waits for the event before proceeding.

The Sending node is untested as we could not test on Keysight's hardware.