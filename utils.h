#pragma once

#include <cstddef>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

const size_t MB = 1024 * 1024;

template <typename T>
void read_safe(std::ifstream &infile, T &dest) {
    infile.read((char *)&dest, sizeof(T));
}

int32_t get_num_codebooks(float bandwidth, int hop_length, float sample_rate) {
    // The number of codebooks is determined by the bandwidth selected.
    // Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8),
    // 12 kbps (n_q = 16) and 24kbps (n_q = 32).
    return (int32_t)ceilf(1000 * bandwidth / (ceilf(sample_rate / hop_length) * 10));
}

int32_t get_bandwidth_per_quantizer(int bins, float frame_rate) {
    return log2f((float)bins) * frame_rate;
}

int32_t get_num_quantizers_for_bandwidth(int bins, float frame_rate, float bandwidth) {
    float bw_per_q = get_bandwidth_per_quantizer(bins, frame_rate);
    int32_t n_q = MAX(1, floorf(bandwidth * 1000 / bw_per_q));
    return n_q;
}
