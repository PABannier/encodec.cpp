#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "encodec.h"
#include "common.h"

struct encodec_params {
    // number of threads for inference
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    // weights location
    std::string model_path = "/Users/pbannier/Documents/encodec.cpp/ggml_weights/ggml-model.bin";

    // input location
    std::string original_audio_path = "/Users/pbannier/Documents/encodec/decomp_24khz_True.wav";

    // output location
    std::string dest_path = "output.bin";
};

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
    fprintf(stderr, "  -o FNAME, --output FNAME\n");
    fprintf(stderr, "                        output compressed audio (default: %s)\n", params.dest_path.c_str());
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
            params.dest_path = argv[++i];
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

int main(int argc, char **argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    encodec_params params;

    if (encodec_params_parse(argc, argv, params) > 0) {
        fprintf(stderr, "%s: Could not parse arguments\n", __func__);
        return 1;
    }

    // initialize encodec context
    struct encodec_context * ectx = encodec_load_model(params.model_path);
    if (!ectx) {
        printf("%s: error during loading model\n", __func__);
        return 1;
    }

    // read audio from disk
    std::vector<float> original_audio_arr;
    if(!read_wav_from_disk(params.original_audio_path, original_audio_arr)) {
        printf("%s: error during reading wav file\n", __func__);
        return 1;
    }

    printf("\n");

    // reconstruct audio
    if (!encodec_reconstruct_audio(ectx, original_audio_arr, params.n_threads)) {
        printf("%s: error during inference\n", __func__);
        return 1;
    }

    // write reconstructed audio on disk
    auto & audio_arr = ectx->out_audio;
    write_wav_on_disk(audio_arr, params.dest_wav_path);

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, ectx->t_load_us/1000.0f);
        printf("%s:     eval time = %8.2f ms\n", __func__, ectx->t_compute_ms/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    encodec_free(ectx);

    return 0;
}