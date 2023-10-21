#include <string>
#include <vector>

struct encodec_params {
    // number of threads for inference
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    // weights location
    std::string model_path = "/Users/pbannier/Documents/encodec.cpp/ggml_weights/ggml-model.bin";

    // input location
    std::string original_audio_path = "/Users/pbannier/Documents/encodec/decomp_24khz_True.wav";

    // output location
    std::string dest_wav_path = "output.wav";
};

int encodec_params_parse(int argc, char ** argv, encodec_params & params);

bool read_wav_from_disk(std::string in_path, std::vector<float>& audio_arr);

void write_wav_on_disk(std::vector<float>& audio_arr, std::string dest_path);