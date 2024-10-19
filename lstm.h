#pragma once

#include <vector>
#include "ggml.h"


typedef std::vector<struct encodec_lstm_layer> encodec_lstm_layers;

struct encodec_lstm_layer {
    struct ggml_tensor * weight_ih;
    struct ggml_tensor * weight_hh;
    struct ggml_tensor * bias_ih;
    struct ggml_tensor * bias_hh;
};

// Forward pass for a LSTM model with one or multiple layers.
// inp: input tensor of shape [n_timesteps, input_dim]
struct ggml_tensor * encodec_lstm(struct ggml_context  * main_ctx,
                                  struct ggml_tensor   * inp,
                                  encodec_lstm_layers    layers);
