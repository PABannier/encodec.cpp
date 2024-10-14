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

    encodec_lstm lstm;

    struct ggml_tensor *final_conv_w;
    struct ggml_tensor *final_conv_b;

    std::vector<encodec_encoder_block> blocks;
};

struct ggml_tensor *encodec_forward_encoder(
    const struct encodec_encoder *encoder, struct ggml_context *ctx0,
    struct ggml_tensor *inp, const int * ratios, const int kernel_size, const int res_kernel_size,
    const int stride) {

    if (!inp) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    struct ggml_tensor *inpL = strided_conv_1d(
        ctx0, inp, encoder->init_conv_w, encoder->init_conv_b, stride);

    for (int layer_ix = 0; layer_ix < 4; layer_ix++) {
        encodec_encoder_block block = encoder->blocks[layer_ix];

        struct ggml_tensor *current = inpL;

        // shortcut
        struct ggml_tensor *shortcut = strided_conv_1d(
            ctx0, inpL, block.conv_sc_w, block.conv_sc_b, stride);

        // conv1
        current = ggml_elu(ctx0, current);

        current = strided_conv_1d(
            ctx0, current, block.conv_1_w, block.conv_1_b, stride);

        // conv2
        current = ggml_elu(ctx0, current);

        current = strided_conv_1d(
            ctx0, current, block.conv_2_w, block.conv_2_b, stride);

        // residual connection
        inpL = ggml_add(ctx0, current, shortcut);

        // downsampling layers
        inpL = ggml_elu(ctx0, inpL);

        inpL = strided_conv_1d(
            ctx0, inpL, block.ds_conv_w, block.ds_conv_b, ratios[3 - layer_ix]);
    }

    // lstm
    {
        struct ggml_tensor *cur = inpL;

        const encodec_lstm lstm = encoder->lstm;

        // first lstm layer
        char l0_prefix[7] = "enc_l0";
        struct ggml_tensor *hs1 = forward_pass_lstm_unilayer(
            ctx0, cur, lstm.l0_ih_w, lstm.l0_hh_w, lstm.l0_ih_b, lstm.l0_hh_b, l0_prefix);

        // second lstm layer
        char l1_prefix[7] = "enc_l1";
        struct ggml_tensor *out = forward_pass_lstm_unilayer(
            ctx0, hs1, lstm.l1_ih_w, lstm.l1_hh_w, lstm.l1_ih_b, lstm.l1_hh_b, l1_prefix);

        inpL = ggml_add(ctx0, inpL, out);
    }

    // final conv
    inpL = ggml_elu(ctx0, inpL);

    struct ggml_tensor *encoded_inp = strided_conv_1d(
        ctx0, inpL, encoder->final_conv_w, encoder->final_conv_b, stride);

    return encoded_inp;
}
