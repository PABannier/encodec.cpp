#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>
#include <string> 
#include <vector>

#include "ggml.h"

#define ENCODEC_FILE_MAGIC   'ggml'
#define ENCODEC_FILE_VERSION 1

static const size_t MB = 1024*1024;

struct encodec_hparams {
    int32_t in_channels          = 1;
    int32_t hidden_dim           = 128;
    int32_t n_filters            = 32;
    int32_t ratios[4]            = {8, 5, 4, 2};
    int32_t kernel_size          = 7;
    int32_t residual_kernel_size = 3;
    int32_t compress             = 2;
    int32_t n_lstm_layers        = 2;
    int32_t stride               = 1;

    // number of codebooks is determined by the bandwidth selected.
    // Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
    int32_t n_q                  = 32;
    int32_t n_bins               = 1024;
    int32_t sr                   = 24000;
};

struct encodec_model;

struct encodec_context {
    encodec_context(encodec_model & model) : model(model) {}

    ~encodec_context() {
        if (model_owner) {
            delete &model;
        }
    }

    encodec_model & model;
    bool model_owner = false;

    struct ggml_context * ctx_audio;
    struct ggml_tensor  * reconstructed_audio;

    // buffer for `ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // buffers to evaluate the model
    std::vector<uint8_t> buf_alloc;
    std::vector<uint8_t> buf_compute;

    struct ggml_allocr * allocr = {};

    // statistics
    int64_t t_compute_ms = 0;
};
