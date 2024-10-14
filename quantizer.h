#pragma once

#include <cassert>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "utils.h"

struct encodec_quant_block {
    struct ggml_tensor *embed;
};

struct encodec_quantizer {
    std::vector<encodec_quant_block> blocks;
};

struct ggml_tensor *encodec_forward_quantizer_encode(
    const struct encodec_quantizer *quantizer, struct ggml_allocr *allocr, struct ggml_context *ctx0,
    struct ggml_tensor *encoded_inp, const int n_bins, const int sr, const int bandwidth,
    const int hop_length) {

    if (!encoded_inp) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    const int frame_rate = (int)ceilf(sr / hop_length);
    const int n_q = get_num_quantizers_for_bandwidth(n_bins, frame_rate, bandwidth);

    const int seq_length = encoded_inp->ne[0];

    struct ggml_tensor *codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, seq_length, n_q);
    ggml_allocr_alloc(allocr, codes);

    struct ggml_tensor *dist_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_allocr_alloc(allocr, dist_scale);

    if (!ggml_allocr_is_measure(allocr)) {
        float s = -2.0f;
        ggml_backend_tensor_set(dist_scale, &s, 0, sizeof(s));
    }

    struct ggml_tensor *inpL = ggml_cont(ctx0, ggml_transpose(ctx0, encoded_inp));
    struct ggml_tensor *residual = inpL;
    struct ggml_tensor *indices;

    for (int i = 0; i < n_q; i++) {
        encodec_quant_block block = quantizer->blocks[i];

        // compute distance
        // [seq_length, n_bins]
        struct ggml_tensor *dp = ggml_scale(
            ctx0, ggml_mul_mat(ctx0, block.embed, residual), dist_scale);

        // [n_bins]
        struct ggml_tensor *sqr_embed = ggml_sqr(ctx0, block.embed);
        struct ggml_tensor *sqr_embed_nrm = ggml_sum_rows(ctx0, sqr_embed);

        // [seq_length]
        struct ggml_tensor *sqr_inp = ggml_sqr(ctx0, residual);
        struct ggml_tensor *sqr_inp_nrm = ggml_sum_rows(ctx0, sqr_inp);

        // [seq_length, n_bins]
        struct ggml_tensor *dist = ggml_add(ctx0, ggml_repeat(ctx0, sqr_inp_nrm, dp), dp);
        dist = ggml_add(ctx0, ggml_repeat(ctx0, ggml_transpose(ctx0, sqr_embed_nrm), dist), dist);
        dist = ggml_neg(ctx0, dist);

        // take the argmax over the column dimension
        // [seq_length]
        indices = ggml_argmax(ctx0, dist);

        // look up in embedding table
        struct ggml_tensor *quantized = ggml_get_rows(ctx0, block.embed, indices);

        residual = ggml_sub(ctx0, residual, quantized);

        codes = ggml_set_1d(ctx0, codes, indices, i * codes->nb[1]);
    }

    return codes;
}

struct ggml_tensor *encodec_forward_quantizer_decode(
    const struct encodec_quantizer *quantizer, struct ggml_allocr *allocr, struct ggml_context *ctx0,
    struct ggml_tensor *codes, const int hidden_dim, const int n_bins, const int sr, const int bandwidth,
    const int hop_length) {

    if (!codes) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    const int seq_length = codes->ne[0];

    const int frame_rate = (int)ceilf(sr / hop_length);
    const int n_q = get_num_quantizers_for_bandwidth(n_bins, frame_rate, bandwidth);

    assert(n_q == codes->ne[1]);

    struct ggml_tensor *quantized_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, seq_length);
    ggml_allocr_alloc(allocr, quantized_out);

    if (!ggml_allocr_is_measure(allocr)) {
        quantized_out = ggml_set_zero(quantized_out);
    }

    for (int i = 0; i < n_q; i++) {
        encodec_quant_block block = quantizer->blocks[i];

        struct ggml_tensor *indices = ggml_view_1d(ctx0, codes, seq_length, i * codes->nb[1]);
        struct ggml_tensor *quantized = ggml_get_rows(ctx0, block.embed, indices);

        quantized_out = ggml_add(ctx0, quantized_out, quantized);
    }

    quantized_out = ggml_cont(ctx0, ggml_transpose(ctx0, quantized_out));

    return quantized_out;
}
