#pragma once

#include <cstddef>
#include <fstream>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

const size_t MB = 1024 * 1024;

template <typename T>
void read_safe(std::ifstream &infile, T &dest) {
    infile.read((char *)&dest, sizeof(T));
}

int32_t get_num_codebooks(float bandwidth, int hop_length, float sample_rate);

int32_t get_bandwidth_per_quantizer(int bins, float frame_rate);

int32_t get_num_quantizers_for_bandwidth(int bins, float frame_rate, float bandwidth);
