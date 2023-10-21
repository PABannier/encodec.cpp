#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "json.hpp"

#include "common.h"

#define SAMPLE_RATE 24000
#define BITS_PER_CODEBOOK 10    // int(log2(quantizer.bins)); quantizer.bins = 1024

#define ENCODEC_MAGIC 'ECDC'


#pragma pack(push, 1)   //  exact fit - no padding
struct encodec_file_header {
    uint32_t magic;
    uint32_t version;
    uint32_t meta_length;
};
#pragma pack(pop)        // back to whatever the previous packing mode was


void encodec_print_usage(char ** argv, const encodec_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model_path.c_str());
    fprintf(stderr, "  -i FNAME, --input FNAME\n");
    fprintf(stderr, "                        original audio wav (default: %s)\n", params.original_audio_path.c_str());
    fprintf(stderr, "  -o FNAME, --outwav FNAME\n");
    fprintf(stderr, "                        output generated wav (default: %s)\n", params.dest_wav_path.c_str());
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
            params.dest_wav_path = argv[++i];
        } else if (arg == "-i" || arg == "--input") {
            params.original_audio_path = argv[++i];
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

bool read_wav_from_disk(std::string in_path, std::vector<float>& audio_arr) {
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

void write_wav_on_disk(std::vector<float>& audio_arr, std::string dest_path) {
    drwav_data_format format;
    format.container     = drwav_container_riff;
    format.format        = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels      = 1;
    format.sampleRate    = SAMPLE_RATE;
    format.bitsPerSample = 32;

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

void write_encodec_header(std::ofstream & fo, std::vector<int32_t> & codes) {
    assert(codes.size() % 32 == 0);  // codes.size() must be a multiple of 32 (32 codebooks)

    uint32_t audio_length = codes.size() / 32;
    nlohmann::json metadata = {
        {"model_name"  , "encodec_24khz"},
        {"audio_length",    audio_length},
        {"n_codebooks" ,              32},
        {"use_lm"      ,           false},
    };
    std::string meta_dumped = metadata.dump();

    encodec_file_header header;
    header.magic = ENCODEC_MAGIC;
    header.version = 0;
    header.meta_length = static_cast<uint32_t>(meta_dumped.size());

    fo.write(reinterpret_cast<char *>(&header), sizeof(header));
    fo.write(meta_dumped.c_str(), meta_dumped.size());
    fo.flush();
}

void write_encodec_codes(std::ofstream & fo, std::vector<int32_t> & codes) {
    BitPacker bp(BITS_PER_CODEBOOK, fo);
    for (int32_t code : codes) {
        bp.push(code);
    }
    bp.flush();
}

bool write_codes_to_file(std::string dest_path, std::vector<int32_t> & codes) {
    std::ofstream fo(dest_path, std::ios::binary);
    write_encodec_header(fo, codes);
    write_encodec_codes(fo, codes);
    fo.close();

    return true;
}
