#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "json.hpp"

#include "common.h"

#define SAMPLE_RATE 24000
#define BITS_PER_CODEBOOK 10    // int(log2(quantizer.bins)); quantizer.bins = 1024


// The ECDC file format expects big endian byte order.
// This function swaps the endianness of a 32-bit integer.
uint32_t swap_endianness(uint32_t value) {
    return ((value & 0x000000FF) << 24) |
           ((value & 0x0000FF00) << 8) |
           ((value & 0x00FF0000) >> 8) |
           ((value & 0xFF000000) >> 24);
}

// This checks if the system is in big-endian or little-endian order.
bool is_big_endian(void) {
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};

    return bint.c[0] == 1;
}

void encodec_print_usage(char ** argv, const encodec_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model_path.c_str());
    fprintf(stderr, "  -i FNAME, --input FNAME\n");
    fprintf(stderr, "                        original audio wav (default: %s)\n", params.input_path.c_str());
    fprintf(stderr, "  -o FNAME, --outwav FNAME\n");
    fprintf(stderr, "                        output generated wav (default: %s)\n", params.output_path.c_str());
    fprintf(stderr, "\n");
}

int encodec_params_parse(int argc, char ** argv, encodec_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model_path = argv[++i];
        } else if (arg == "-o" || arg == "--outwav") {
            params.output_path = argv[++i];
        } else if (arg == "-i" || arg == "--input") {
            params.input_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            encodec_print_usage(argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            encodec_print_usage(argv, params);
            exit(0);
        }
    }

    return 0;
}

bool read_wav_from_disk(std::string in_path, std::vector<float> & audio_arr) {
    uint32_t channels;
    uint32_t sample_rate;
    drwav_uint64 total_frame_count;

    float * raw_audio = drwav_open_file_and_read_pcm_frames_f32(
        in_path.c_str(), &channels, &sample_rate, &total_frame_count, NULL);

    if (raw_audio == NULL) {
        fprintf(stderr, "%s: could not read wav file\n", __func__);
        return false;
    }

    fprintf(stderr, "\n%s: Number of frames read = %lld.\n", __func__, total_frame_count);

    audio_arr.resize(total_frame_count);
    memcpy(audio_arr.data(), raw_audio, total_frame_count * sizeof(float));

    drwav_free(raw_audio, NULL);

    return true;
}

void write_wav_on_disk(std::vector<float> & audio_arr, std::string dest_path) {
    drwav_data_format format;
    format.bitsPerSample = 32;
    format.sampleRate = SAMPLE_RATE;
    format.container = drwav_container_riff;
    format.channels = 1;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;

    drwav wav;
    drwav_init_file_write(&wav, dest_path.c_str(), &format, NULL);
    drwav_uint64 frames = drwav_write_pcm_frames(&wav, audio_arr.size(), audio_arr.data());
    drwav_uninit(&wav);

    fprintf(stderr, "%s: Number of frames written = %lld.\n", __func__, frames);
}

class BitPacker {
    public:
        // Constructor
        BitPacker(int bits, std::ofstream& fo)
            : current_value(0), current_bits(0), bits(bits), fo(fo) {}

        // Member function to push a new value to the stream
        void push(int value) {
            current_value += (value << current_bits);
            current_bits += bits;
            while (current_bits >= 8) {
                uint8_t lower_8bits = current_value & 0xff;
                current_bits -= 8;
                current_value >>= 8;
                fo.write(reinterpret_cast<char*>(&lower_8bits), sizeof(lower_8bits));
            }
        }

        // Member function to flush the remaining partial uint8
        void flush() {
            if (current_bits) {
                fo.write(reinterpret_cast<char*>(&current_value), sizeof(uint8_t));
                current_value = 0;
                current_bits = 0;
            }
            fo.flush();
        }

    private:
        int current_value;
        int current_bits;
        int bits;
        std::ofstream & fo;
};

class BitUnpacker {
    public:
        // Constructor
        BitUnpacker(int bits, std::ifstream& fo)
            : bits(bits), fo(fo), mask((1 << bits) - 1), current_value(0), current_bits(0) {}

        // Member function to pull a single value from the stream
        int pull() {
            while (current_bits < bits) {
                char buf;
                if (!fo.read(&buf, 1)) {
                    return {};  // returns empty optional indicating end of stream
                }
                uint8_t character = static_cast<uint8_t>(buf);
                current_value += character << current_bits;
                current_bits += 8;
            }

            int out = current_value & mask;
            current_value >>= bits;
            current_bits -= bits;
            return out;  // returns the extracted value
        }

    private:
        int bits;
        std::ifstream& fo;
        int mask;
        int current_value;
        int current_bits;
};

std::vector<char> read_exactly(std::ifstream & fo, size_t size) {
    std::vector<char> buf;
    buf.reserve(size);

    while (buf.size() < size) {
        char chunk[size];
        fo.read(chunk, size);
        size_t bytesRead = fo.gcount();
        if (bytesRead == 0) {
            throw std::runtime_error("Impossible to read enough data from the stream, " +
                                     std::to_string(size) + " bytes remaining.");
        }
        buf.insert(buf.end(), chunk, chunk + bytesRead);
        size -= bytesRead;
    }

    return buf;
}

void write_encodec_header(std::ofstream & fo, uint32_t audio_length) {
    nlohmann::json metadata = {
        {"m" , "encodec_24khz"},
        {"al",    audio_length},
        {"nc",              16},
        {"lm",           false},
    };
    std::string meta_dumped = metadata.dump();

    char magic[4] = { 'E', 'C', 'D', 'C'};
    uint8_t version = 0;

    uint32_t meta_length = static_cast<uint32_t>(meta_dumped.size());
    if (!is_big_endian()) {
        // if little endian, needs to swap to big-endian order for correct ECDC format.
        meta_length = swap_endianness(meta_length);
    }

    fo.write(magic, 4);
    fo.write(reinterpret_cast<char*>(&version), sizeof(version));
    fo.write(reinterpret_cast<char*>(&meta_length), sizeof(uint32_t));

    fo.write(meta_dumped.data(), meta_dumped.size());
    fo.flush();
}

nlohmann::json read_ecdc_header(std::ifstream& fo) {
    int size_header = 4 * sizeof(char) + sizeof(uint8_t) + sizeof(uint32_t);
    std::vector<char> header_bytes = read_exactly(fo, size_header);

    char * magic = reinterpret_cast<char *>(
        header_bytes.data(), header_bytes.data() + 4*sizeof(char));

    uint8_t * version = reinterpret_cast<uint8_t *>(
        header_bytes.data() + 4*sizeof(char));

    uint32_t * meta_length = reinterpret_cast<uint32_t *>(
        header_bytes.data() + 4*sizeof(char) + sizeof(uint8_t));

    // switch to little endian if necessary
    if (!is_big_endian()) {
        *meta_length = swap_endianness(*meta_length);
    }

    if (strcmp(magic, "ECDC") != 0) {
        throw std::runtime_error("File is not in ECDC format.");
    }

    if (*version != 0) {
        throw std::runtime_error("Version not supported.");
    }

    std::vector<char> meta_bytes = read_exactly(fo, *meta_length);
    std::string meta_str(meta_bytes.begin(), meta_bytes.end());
    return nlohmann::json::parse(meta_str);
}

void write_encodec_codes(
                 std::ofstream & fo,
          std::vector<int32_t> & codes) {
    BitPacker bp(BITS_PER_CODEBOOK, fo);

    for (int32_t code : codes) {
        bp.push(code);
    }

    bp.flush();
}

bool write_codes_to_file(
                   std::string   dest_path,
          std::vector<int32_t> & codes,
                      uint32_t   audio_length) {
    std::ofstream fo(dest_path, std::ios::binary);

    write_encodec_header(fo, audio_length);
    write_encodec_codes(fo, codes);

    fo.close();

    return true;
}

bool read_codes_from_file(
                   std::string   code_path,
          std::vector<int32_t> & codes) {
    std::ifstream fin(code_path, std::ios::binary);

    nlohmann::json metadata = read_ecdc_header(fin);

    uint32_t audio_length = metadata["audio_length"];
    uint32_t n_codebooks  = metadata["n_codebooks"];

    codes.resize(audio_length * n_codebooks);

    BitUnpacker bu(BITS_PER_CODEBOOK, fin);

    for (int i = 0; i < (int) codes.size(); i++) {
        codes[i] = bu.pull();
    }

    fin.close();

    return true;
}
