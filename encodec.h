/*
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2024 Pierre-Antoine Bannier                                        │
│                                                                              │
│ Permission to use, copy, modify, and/or distribute this software for         │
│ any purpose with or without fee is hereby granted, provided that the         │
│ above copyright notice and this permission notice appear in all copies.      │
│                                                                              │
│ THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL                │
│ WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED                │
│ WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE             │
│ AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL         │
│ DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR        │
│ PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER               │
│ TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR             │
│ PERFORMANCE OF THIS SOFTWARE.                                                │
╚─────────────────────────────────────────────────────────────────────────────*/
/*
 * This file contains the declarations of the structs and functions used in the encodec library.
 * The library provides functionality for audio compression and decompression using a custom model.
 * The model consists of an encoder, a quantizer and a decoder, each with their own set of parameters.
 * The library also provides functions for loading and freeing the model, as well as compressing and decompressing audio data.
 *
 */
#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif
    struct encodec_context;

    struct encodec_statistics {
        // The time taken to load the model.
        int64_t t_load_us;
        // The time taken to compute the model.
        int64_t t_compute_us;
    };

    /**
     * Loads an encodec model from the specified file path.
     *
     * @param model_path The file path to the encodec model.
     * @param offset The offset (in bytes) to the start of the model in the file.
     * @param n_gpu_layers The number of GPU layers to use.
     * @return A pointer to the encodec context struct.
     */
    struct encodec_context *encodec_load_model(
        const char *model_path,
        const int offset,
        int n_gpu_layers);

    /**
     * Sets the target bandwidth for the given encodec context.
     *
     * @param ectx The encodec context to set the target bandwidth for.
     * @param bandwidth The target bandwidth to set, in bits per second.
     */
    void encodec_set_target_bandwidth(
        struct encodec_context *ectx,
        int bandwidth);

    /**
     * Sets the sample rate for the given encodec context.
     *
     * @param ectx The encodec context to set the target bandwidth for.
     * @param sample_rate The sample rate to set.
     */
    void encodec_set_sample_rate(
        struct encodec_context *ectx,
        int sample_rate);

    /**
     * Reconstructs audio from raw audio data using the specified encodec context.
     *
     * @param ectx The encodec context to use for reconstruction.
     * @param raw_audio The raw audio data to reconstruct.
     * @param n_samples The number of samples in the raw audio buffer.
     * @param n_threads The number of threads to use for reconstruction.
     * @return True if the reconstruction was successful, false otherwise.
     */
    bool encodec_reconstruct_audio(
        struct encodec_context *ectx,
        const float *raw_audio,
        const int n_samples,
        int n_threads);

    /**
     * Compresses audio data using the specified encodec context.
     *
     * @param ectx The encodec context to use for compression.
     * @param raw_audio The raw audio data to compress.
     * @param n_samples The number of samples in the raw audio buffer.
     * @param n_threads The number of threads to use for compression.
     * @return True if the compression was successful, false otherwise.
     */
    bool encodec_compress_audio(
        struct encodec_context *ectx,
        const float *raw_audio,
        const int n_samples,
        int n_threads);

    /**
     * Decompresses audio data using the specified encodec context.
     *
     * @param ectx The encodec context to use for decompression.
     * @param codes The compressed audio data to decompress.
     * @param n_codes The number of codes in the codes buffer.
     * @param n_threads The number of threads to use for decompression.
     * @return True if the audio data was successfully decompressed, false otherwise.
     */
    bool encodec_decompress_audio(
        struct encodec_context *ectx,
        const int32_t *codes,
        const int n_codes,
        int n_threads);

    /**
     * Gets the audio data from the given encodec context.
     *
     * @param ectx The encodec context to get the audio data from.
     * @return A pointer to the audio data.
    */
    float * encodec_get_audio(
        struct encodec_context *ectx);

    /**
     * Gets the size of the audio data from the given encodec context.
     *
     * @param ectx The encodec context to get the audio size from.
     * @return The size of the audio data.
    */
    int encodec_get_audio_size(
        struct encodec_context *ectx);

    /**
     * Gets the code data from the given encodec context.
     *
     * @param ectx The encodec context to get the code data from.
     * @return A pointer to the code data.
    */
    int32_t * encodec_get_codes(
        struct encodec_context *ectx);

    /**
     * Gets the size of the code data from the given encodec context.
     *
     * @param ectx The encodec context to get the code size from.
     * @return The size of the code data.
    */
    int encodec_get_codes_size(
        struct encodec_context *ectx);

    /**
     * Gets the statistics for the given encodec context.
     *
     * @param ectx The encodec context to get the statistics for.
     * @return A pointer to the statistics struct.
    */
    const struct encodec_statistics* encodec_get_statistics(
        struct encodec_context *ectx);

    /**
     * Reset the statistics for the given encodec context.
     *
     * @param ectx The encodec context to reset the statistics for.
    */
   void encodec_reset_statistics(
        struct encodec_context *ectx);

    /**
     * @brief Frees the memory allocated for an encodec context.
     *
     * @param ectx The encodec context to free.
     */
    void encodec_free(
        struct encodec_context *ectx);

#ifdef __cplusplus
}
#endif