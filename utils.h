#pragma once

#include <cstddef>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

const size_t MB = 1024 * 1024;

template <typename T>
static void read_safe(std::ifstream &infile, T &dest) {
    infile.read((char *)&dest, sizeof(T));
}
