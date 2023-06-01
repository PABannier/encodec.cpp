#pragma once

#include "ggml.h"

struct ggml_tensor * strided_conv_1d(
            ggml_context * ctx0,
             ggml_tensor * inp,
             ggml_tensor * conv_w_v,
             ggml_tensor * conv_w_g,
             ggml_tensor * conv_b,
                     int   stride);
