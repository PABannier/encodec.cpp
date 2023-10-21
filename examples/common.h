#include <string>
#include <vector>


bool read_wav_from_disk(std::string in_path, std::vector<float>& audio_arr);

void write_wav_on_disk(std::vector<float>& audio_arr, std::string dest_path);