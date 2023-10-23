#include <string>
#include <vector>

/**
 * @brief Struct containing parameters for the encodec context.
 *
 */
struct encodec_params {
    // number of threads for inference
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    // weights location
    std::string model_path = "/Users/pbannier/Documents/encodec.cpp/ggml_weights/ggml-model.bin";

    // input location
    std::string input_path = "/Users/pbannier/Documents/encodec/decomp_24khz_True.wav";

    // output location
    std::string output_path = "output.wav";

    // number of GPU layers to use
    int32_t n_gpu_layers = 0;
};

/**
 * @brief Parses command line arguments and sets the encodec parameters accordingly.
 *
 * @param argc Number of command line arguments.
 * @param argv Array of command line arguments.
 * @param params Struct containing encodec parameters.
 * @return int Returns 0 if successful, -1 otherwise.
 */
int encodec_params_parse(int argc, char ** argv, encodec_params & params);

/**
 * @brief Reads a WAV file from disk and stores the audio data in a vector of floats.
 *
 * @param in_path Path to the input WAV file.
 * @param audio_arr Vector to store the audio data.
 * @return true If the file was successfully read.
 * @return false If the file could not be read.
 */
bool read_wav_from_disk(std::string in_path, std::vector<float> & audio_arr);

/**
 * @brief Writes a vector of floats to a WAV file on disk.
 *
 * @param audio_arr Vector containing the audio data.
 * @param dest_path Path to the output WAV file.
 */
void write_wav_on_disk(std::vector<float>& audio_arr, std::string dest_path);

/**
 * @brief Writes a vector of integers to a file on disk.
 *
 * @param dest_path Path to the output file.
 * @param codes Vector containing the integers to write.
 * @param audio_length Original length of the audio.
 * @return true If the file was successfully written.
 * @return false If the file could not be written.
 */
bool write_codes_to_file(std::string dest_path, std::vector<int32_t> & codes, uint32_t audio_length);

/**
 * @brief Reads a vector of integers from a file on disk.
 *
 * @param code_path Path to the input file.
 * @param codes Vector to store the codes.
 * @param audio_length Original length of the audio.
 * @param n_codebooks Number of codebooks used to encode the audio.
 * @return std::vector<int32_t> Vector containing the integers read from the file.
 */
bool read_codes_from_file(
                   std::string   code_path,
          std::vector<int32_t> & codes,
                      uint32_t & audio_length,
                      uint32_t & n_codebooks);
