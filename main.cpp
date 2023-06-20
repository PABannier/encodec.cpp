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

    // generate toy data
    std::vector<float> raw_audio(1000, 0.4);

    // encode
    const int64_t t_compr_us_start = ggml_time_us();

    // TODO change n_threads to be passed by params
    encodec_model_eval(raw_audio, model, 1);

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