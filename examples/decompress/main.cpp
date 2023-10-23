#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "encodec.h"
#include "common.h"


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

    encodec_set_target_bandwidth(ectx, 12);

    // read compressed audio from disk
    std::vector<int32_t> codes;
    uint32_t audio_length, n_codebooks;
    if (!read_codes_from_file(params.input_path, codes, audio_length, n_codebooks)) {
        printf("%s: error during reading codes\n", __func__);
        return 1;
    }

    // decompress audio
    if (!encodec_decompress_audio(ectx, codes, params.n_threads)) {
        printf("%s: error during decompression\n", __func__);
        return 1;
    }

    // write reconstructed audio on disk
    auto & audio_arr = ectx->out_audio;
    audio_arr.resize(audio_length);
    write_wav_on_disk(audio_arr, params.output_path);

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
