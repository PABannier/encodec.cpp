#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#define ENCODEC_FILE_MAGIC   'ggml'

static const size_t MB = 1024*1024;

struct encodec_hparams {
    // The number of input channels is always 1 (mono).
    int32_t in_channels          = 1;
    // The hidden dimension for the codebook.
    int32_t hidden_dim           = 128;
    // The number of filters for the first convolution.
    int32_t n_filters            = 32;
    // The filter size for upsampling and downsampling.
    int32_t ratios[4]            = {8, 5, 4, 2};
    // The kernel size for the first convolution.
    int32_t kernel_size          = 7;
    // The kernel size for the residual blocks.
    int32_t residual_kernel_size = 3;
    // Compression
    int32_t compress             = 2;
    // The number of layers in the LSTM modules.
    int32_t n_lstm_layers        = 2;
    // The stride of the first convolution.
    int32_t stride               = 1;

    // The number of codebooks is determined by the bandwidth selected.
    // Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8),
    // 12 kbps (n_q = 16) and 24kbps (n_q = 32).
    int32_t n_q    = 32;
    int32_t n_bins = 1024;
    int32_t sr     = 24000;

    int32_t ftype;
};

// res + downsample block at some ratio
struct encodec_encoder_block {
    // conv1
    struct ggml_tensor * conv_1_w;
    struct ggml_tensor * conv_1_b;

    // conv2
    struct ggml_tensor * conv_2_w;
    struct ggml_tensor * conv_2_b;

    // shortcut
    struct ggml_tensor * conv_sc_w;
    struct ggml_tensor * conv_sc_b;

    // downsampling layers
    struct ggml_tensor * ds_conv_w;
    struct ggml_tensor * ds_conv_b;
};

struct encodec_lstm {
    struct ggml_tensor * l0_ih_w;
    struct ggml_tensor * l0_hh_w;

    struct ggml_tensor * l0_ih_b;
    struct ggml_tensor * l0_hh_b;

    struct ggml_tensor * l1_ih_w;
    struct ggml_tensor * l1_hh_w;

    struct ggml_tensor * l1_ih_b;
    struct ggml_tensor * l1_hh_b;
};

struct encodec_encoder {
    struct ggml_tensor * init_conv_w;
    struct ggml_tensor * init_conv_b;

    encodec_lstm lstm;

    struct ggml_tensor * final_conv_w;
    struct ggml_tensor * final_conv_b;

    std::vector<encodec_encoder_block> blocks;
};

struct encodec_quant_block {
    struct ggml_tensor * embed;
};

struct encodec_quantizer {
    std::vector<encodec_quant_block> blocks;
};

struct encodec_decoder_block {
    //upsampling layers
    struct ggml_tensor * us_conv_w;
    struct ggml_tensor * us_conv_b;

    // conv1
    struct ggml_tensor * conv_1_w;
    struct ggml_tensor * conv_1_b;

    // conv2
    struct ggml_tensor * conv_2_w;
    struct ggml_tensor * conv_2_b;

    // shortcut
    struct ggml_tensor * conv_sc_w;
    struct ggml_tensor * conv_sc_b;
};

struct encodec_decoder {
    struct ggml_tensor * init_conv_w;
    struct ggml_tensor * init_conv_b;

    encodec_lstm lstm;

    struct ggml_tensor * final_conv_w;
    struct ggml_tensor * final_conv_b;

    std::vector<encodec_decoder_block> blocks;
};

struct encodec_model {
    encodec_hparams hparams;

    encodec_encoder   encoder;
    encodec_quantizer quantizer;
    encodec_decoder   decoder;

    // context
    struct ggml_context * ctx;
    int n_loaded;

    ggml_backend_t backend = NULL;

    ggml_backend_buffer_t buffer_w;

    std::map<std::string, struct ggml_tensor *> tensors;
};

struct encodec_context {
    encodec_model model;

    // buffer for model evaluation
    ggml_backend_buffer_t buf_compute;

    // custom allocrator
    struct ggml_allocr * allocr = NULL;

    // output audio
    std::vector<float> out_audio;

    // statistics
    int64_t t_load_us    = 0;
    int64_t t_compute_ms = 0;
};

struct encodec_context * encodec_load_model(const std::string & model_path);

bool encodec_reconstruct_audio(
            struct encodec_context * ectx,
                std::vector<float> & raw_audio,
                               int   n_threads);

void encodec_free(struct encodec_context * ectx);