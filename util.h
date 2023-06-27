#pragma once

#include <fstream>

#define ENCODEC_FILE_MAGIC   'ggml'
#define ENCODEC_FILE_VERSION 1


#define ENCODEC_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "ENCODEC_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

static const size_t MB = 4*1024*1024;

template<typename T>
static void read_safe(std::ifstream& infile, T& dest) {
    infile.read((char*)& dest, sizeof(T));
}
