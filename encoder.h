#pragma once

#include "ggml.h"

struct ggml_tensor * strided_conv_1d(
      ggml_context * ctx0,
       ggml_tensor * inp,
       ggml_tensor * conv_w,
       ggml_tensor * conv_b,
               int   stride);

struct ggml_tensor * forward_pass_lstm_unilayer(
      ggml_context * ctx0,
       ggml_tensor * inp,
       ggml_tensor * weight_ih,
       ggml_tensor * weight_hh,
       ggml_tensor * bias_ih,
       ggml_tensor * bias_hh);
