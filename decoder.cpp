/* Implementation of the decoder part of the Encodec model.

For more details, see the explanation in the encoder.cpp file.
*/

#include "ggml.h"

#include "decoder.h"
#include "lstm.h"
#include "ops.h"
#include "utils.h"

const static int DECODER_TOTAL_NUM_NODES = 220;

static struct ggml_tensor *encodec_forward_decoder_step_0(
    const struct encodec_decoder * decoder,
             struct ggml_context * ctx0,
              struct ggml_tensor * quantized_out,
                       const int   stride) {

    struct ggml_tensor *inpL = strided_conv_1d(
        ctx0, quantized_out, decoder->init_conv_w, decoder->init_conv_b, stride);

    return inpL;
}

static struct ggml_tensor *encodec_forward_decoder_step_1(
    const struct encodec_decoder * decoder,
             struct ggml_context * ctx0,
              struct ggml_tensor * inpL,
                       const int * ratios,
                       const int   stride,
                       const int   kernel_size,
                       const int   res_kernel_size) {

    struct ggml_tensor *cur = inpL;

    // multi-layer lstm
    struct ggml_tensor *out = encodec_lstm(ctx0, cur, decoder->lstm_layers);

    inpL = ggml_add(ctx0, inpL, out);

    for (int layer_ix = 0; layer_ix < 4; layer_ix++) {
        encodec_decoder_block block = decoder->blocks[layer_ix];

        // upsampling layers
        inpL = ggml_elu(ctx0, inpL);

        inpL = strided_conv_transpose_1d(
            ctx0, inpL, block.us_conv_w, block.us_conv_b, ratios[layer_ix]);

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
    }

    // final conv
    inpL = ggml_elu(ctx0, inpL);

    struct ggml_tensor *decoded_inp = strided_conv_1d(
        ctx0, inpL, decoder->final_conv_w, decoder->final_conv_b, stride);

    return decoded_inp;
}

struct ggml_tensor *encodec_forward_decoder(
    const struct encodec_decoder * decoder,
             struct ggml_context * main_ctx,
              struct ggml_tensor * quantized_out,
                       const int * ratios,
                       const int   kernel_size,
                       const int   res_kernel_size,
                       const int   stride) {
    // quantized_out lives in ctx0
    if (!quantized_out) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    // setup decoder context
    static size_t buf_size = ggml_tensor_overhead() * DECODER_TOTAL_NUM_NODES + ggml_graph_overhead();
    buf_size += 1024 * 1024 * 1024; // 1 MB (extra safety margin)

    struct ggml_init_params params = {
        /* .mem_size   = */ buf_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * decoder_ctx = ggml_init(params);
    struct ggml_cgraph  * gf = ggml_new_graph(decoder_ctx);

    // step 0
    struct ggml_tensor * inpL = encodec_forward_decoder_step_0(decoder, decoder_ctx, quantized_out, stride);
    ggml_set_output(inpL);

    ggml_build_forward_expand(gf, inpL);
    printf("[decoder] number of nodes: %d\n", ggml_graph_n_nodes(gf));
    ggml_graph_compute_with_ctx(decoder_ctx, gf, 4 /* num_threads */);

    // step 1
    struct ggml_tensor * out = encodec_forward_decoder_step_1(decoder, decoder_ctx, inpL, ratios, stride, kernel_size, res_kernel_size);
    ggml_set_output(out);

    ggml_build_forward_expand(gf, out);
    printf("[decoder] number of nodes: %d\n", ggml_graph_n_nodes(gf));
    ggml_graph_compute_with_ctx(decoder_ctx, gf, 4 /* num_threads */);

    // copy output to main context
    struct ggml_tensor * decoded = ggml_new_tensor_2d(main_ctx, GGML_TYPE_F32, out->ne[0], out->ne[1]);
    memcpy(decoded->data, out->data, ggml_nbytes(out));
    ggml_set_name(decoded, "decoded");

    ggml_free(decoder_ctx);

    return decoded;
}