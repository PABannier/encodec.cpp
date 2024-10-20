#pragma once

#include "ggml.h"

struct ggml_tensor *pad_1d(struct ggml_context *ctx0, struct ggml_tensor *inp,
                           int padding_left, int padding_right);

struct ggml_tensor *unpad_1d(struct ggml_context *ctx0, struct ggml_tensor *inp,
                             int padding_left, int padding_right);

struct ggml_tensor *strided_conv_1d(struct ggml_context *ctx0, struct ggml_tensor *inp,
                                    struct ggml_tensor *conv_w, struct ggml_tensor *conv_b,
                                    int stride);

struct ggml_tensor *strided_conv_transpose_1d(struct ggml_context *ctx0, struct ggml_tensor *inp,
                                              struct ggml_tensor *conv_w, struct ggml_tensor *conv_b,
                                              int stride);
