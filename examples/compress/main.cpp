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
    struct encodec_context * ectx = encodec_load_model(params.model_path.c_str(), 0 /* offset */, params.n_gpu_layers);
    if (!ectx) {
        printf("%s: error during loading model\n", __func__);
        return 1;
    }

    encodec_set_target_bandwidth(ectx, 12);

    // read audio from disk
    std::vector<float> original_audio_arr;
    if(!read_wav_from_disk(params.input_path, original_audio_arr)) {
        printf("%s: error during reading wav file\n", __func__);
        return 1;
    }

    // compress audio
    float * audio_data = original_audio_arr.data();
    int n_samples = original_audio_arr.size();
    if (!encodec_compress_audio(ectx, audio_data, n_samples, params.n_threads)) {
        printf("%s: error during compression \n", __func__);
        return 1;
    }

    // write reconstructed audio on disk
    int32_t * codes_data = encodec_get_codes(ectx);
    int n_codes = encodec_get_codes_size(ectx);
    std::vector<int32_t> codes_arr(codes_data, codes_data + n_codes);

    if (!write_codes_to_file(params.output_path, codes_arr, original_audio_arr.size())) {
        printf("%s: error during writing codes to file\n", __func__);
        return 1;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        const encodec_statistics * stats = encodec_get_statistics(ectx);

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, stats->t_load_us/1000.0f);
        printf("%s:     eval time = %8.2f ms\n", __func__, stats->t_compute_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    encodec_free(ectx);

    return 0;
}