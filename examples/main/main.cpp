#include <cstring>
#include <string>
#include <thread>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "encodec.h"

#define SAMPLE_RATE 24000

struct encodec_params {
    // number of threads for inference
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    // weights location
    std::string model_path = "./ggml_weights";

    // input location
    std::string original_audio_path = "./input.wav";

    // output location
    std::string dest_wav_path = "output.wav";
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
    fprintf(stderr, "                        original audio wav (default: %s)\n", params.dest_wav_path.c_str());
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

    fprintf(stderr, "Number of frames read = %lld.\n", total_frame_count);

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

struct encodec_context encodec_init_from_params(encodec_params & params) {
    encodec_model model = encodec_load_model_from_file(params.model_path);
    encodec_context ectx = encodec_new_context_with_model(model);

    return ectx;
}

int main(int argc, char **argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    encodec_params params;

    if (encodec_params_parse(argc, argv, params) > 0) {
        fprintf(stderr, "%s: Could not parse arguments\n", __func__);
        return 1;
    }

    int64_t t_load_us = 0;
    int64_t t_eval_us = 0;

    // initialize encodec context
    const int64_t t_start_us = ggml_time_us();
    encodec_context ectx = encodec_init_from_params(params);
    t_load_us = ggml_time_us() - t_start_us;

    printf("\n");

    // read audio from disk
    std::vector<float> original_audio_arr;
    if(!read_wav_from_disk(params.original_audio_path, original_audio_arr)) {
        printf("%s: error during reading wav file\n", __func__);
        return 1;
    }

    // reconstruct audio
    const int64_t t_eval_us_start = ggml_time_us();
    if (!encodec_model_eval(ectx, original_audio_arr, params.n_threads)) {
        printf("%s: error during inference\n", __func__);
        return 1;
    }
    t_eval_us = ggml_time_us() - t_eval_us_start;

    // write reconstructed audio on disk
    std::vector<float> audio_arr(ectx.reconstructed_audio->ne[0]);
    memcpy(ectx.reconstructed_audio->data, audio_arr.data(), audio_arr.size() * sizeof(float));
    write_wav_on_disk(audio_arr, params.dest_wav_path);

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:     eval time = %8.2f ms\n", __func__, t_eval_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    encodec_free(ectx);

    return 0;
}