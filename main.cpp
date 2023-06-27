#include "common.h"
#include "encodec.h"
#include "ggml.h"

#include <string>
#include <vector>


int main(int argc, char **argv) {
    encodec_params params;

    if (encodec_params_parse(argc, argv, params) == false) {
        return 1;
    }

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_load_us  = 0;
    int64_t t_compr_us = 0;

    encodec_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!encodec_model_load(params.model, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    // load audio file
    std::vector<float> pcmf32;               // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

    params.in_audio_path = "./test_24k.wav";

    if (!read_wav(params.in_audio_path, pcmf32, pcmf32s, false)) {
        fprintf(stderr, "error: failed to read WAV file '%s'\n", params.in_audio_path.c_str());
        return 1;
    }

    // encode
    const int64_t t_compr_us_start = ggml_time_us();

    // TODO change n_threads to be passed by params
    encodec_model_eval(pcmf32, model, 1);

    t_compr_us = ggml_time_us() - t_compr_us_start;

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s: compress time = %8.2f ms\n", __func__, t_compr_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}