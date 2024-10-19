#pragma once

#include <vector>

#include "ggml.h"
#include "lstm.h"

// res + downsample block at some ratio
struct encodec_encoder_block {
    // conv1
    struct ggml_tensor *conv_1_w;
    struct ggml_tensor *conv_1_b;

    // conv2
    struct ggml_tensor *conv_2_w;
    struct ggml_tensor *conv_2_b;

    // shortcut
    struct ggml_tensor *conv_sc_w;
    struct ggml_tensor *conv_sc_b;

    // downsampling layers
    struct ggml_tensor *ds_conv_w;
    struct ggml_tensor *ds_conv_b;
};

struct encodec_encoder {
    struct ggml_tensor *init_conv_w;
    struct ggml_tensor *init_conv_b;

    encodec_lstm_layers lstm_layers;

    struct ggml_tensor *final_conv_w;
    struct ggml_tensor *final_conv_b;

    std::vector<encodec_encoder_block> blocks;
};

struct ggml_tensor *encodec_forward_encoder(
    const struct encodec_encoder * encoder,
             struct ggml_context * main_ctx,
              struct ggml_tensor * inp,
                       const int * ratios,
                       const int   kernel_size,
                       const int   res_kernel_size,
                       const int   stride);
