#pragma once

#include <vector>

#include "ggml.h"

#include "lstm.h"
#include "ops.h"
#include "utils.h"


struct encodec_decoder_block {
    // upsampling layers
    struct ggml_tensor *us_conv_w;
    struct ggml_tensor *us_conv_b;

    // conv1
    struct ggml_tensor *conv_1_w;
    struct ggml_tensor *conv_1_b;

    // conv2
    struct ggml_tensor *conv_2_w;
    struct ggml_tensor *conv_2_b;

    // shortcut
    struct ggml_tensor *conv_sc_w;
    struct ggml_tensor *conv_sc_b;
};

struct encodec_decoder {
    struct ggml_tensor *init_conv_w;
    struct ggml_tensor *init_conv_b;

    encodec_lstm_layers lstm_layers;

    struct ggml_tensor *final_conv_w;
    struct ggml_tensor *final_conv_b;

    std::vector<encodec_decoder_block> blocks;
};

struct ggml_tensor *encodec_forward_decoder(
    const struct encodec_decoder * decoder,
             struct ggml_context * main_ctx,
              struct ggml_tensor * quantized_out,
                       const int * ratios,
                       const int   kernel_size,
                       const int   res_kernel_size,
                       const int   stride);
