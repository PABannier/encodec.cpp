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

    // read audio from disk
    std::vector<float> original_audio_arr;
    if(!read_wav_from_disk(params.input_path, original_audio_arr)) {
        printf("%s: error during reading wav file\n", __func__);
        return 1;
    }

    // compress audio
    if (!encodec_compress_audio(ectx, original_audio_arr, params.n_threads)) {
        printf("%s: error during compression \n", __func__);
        return 1;
    }

    // write reconstructed audio on disk
    if (!write_codes_to_file(params.output_path, ectx->out_codes, original_audio_arr.size())) {
        printf("%s: error during writing codes to file\n", __func__);
        return 1;
    }

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