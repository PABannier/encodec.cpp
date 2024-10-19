/* Implementation of the Encodec encoder model.

Since the LSTM model must be run in a different context and computational graph than
the rest of the model (see explanations in lstm.cpp), the forward pass of the encoder
is split into 2 computational graphs:

1. The encoder model is run on the input signal until the output of the last encoder
block. This final output serves as the input for the LSTM model.

2. Once the LSTM model has run, a second graph is run for the final convolution.

The `main_ctx` argument is the context of the output tensor of the encoder.
*/
#include "encoder.h"

#include "ggml.h"
#include "lstm.h"
#include "ops.h"

const static int ENCODER_TOTAL_NUM_NODES = 236;

static struct ggml_tensor *encodec_forward_encoder_step_0(
    const struct encodec_encoder * encoder,
             struct ggml_context * ctx0,
              struct ggml_tensor * inp,
                       const int * ratios,
                       const int   kernel_size,
                       const int   res_kernel_size,
                       const int   stride) {

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

    return inpL;
}

static struct ggml_tensor *encodec_forward_encoder_step_1(
    const struct encodec_encoder * encoder,
             struct ggml_context * ctx1,
              struct ggml_tensor * inpL,
                       const int   stride) {
    struct ggml_tensor *cur = inpL;

    struct ggml_tensor *out = encodec_lstm(ctx1, cur, encoder->lstm_layers);

    inpL = ggml_add(ctx1, inpL, out);

    inpL = ggml_elu(ctx1, inpL);

    struct ggml_tensor *output = strided_conv_1d(
        ctx1, inpL, encoder->final_conv_w, encoder->final_conv_b, stride);

    return output;
}

struct ggml_tensor *encodec_forward_encoder(
    const struct encodec_encoder * encoder,
             struct ggml_context * main_ctx,
              struct ggml_tensor * inp,
                       const int * ratios,
                       const int   kernel_size,
                       const int   res_kernel_size,
                       const int   stride) {
    // inp lives in main_ctx
    if (!inp) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    // setup encoder context
    static size_t buf_size = ggml_tensor_overhead() * ENCODER_TOTAL_NUM_NODES;
    buf_size += ggml_graph_overhead();
    buf_size += 1024 * 1024 * 1024; // 1 MB (extra safety margin)

    struct ggml_init_params params = {
        /* .mem_size   = */ buf_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * encoder_ctx = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(encoder_ctx);

    // step 0
    struct ggml_tensor * inpL = encodec_forward_encoder_step_0(encoder, encoder_ctx, inp, ratios, kernel_size, res_kernel_size, stride);
    ggml_set_output(inpL);

    ggml_build_forward_expand(gf, inpL);
    printf("[encoder] number of nodes: %d\n", ggml_graph_n_nodes(gf));
    ggml_graph_compute_with_ctx(encoder_ctx, gf, 4 /* num_threads */);

    // step 1
    struct ggml_tensor * out = encodec_forward_encoder_step_1(encoder, encoder_ctx, inpL, stride);
    ggml_set_output(out);

    ggml_build_forward_expand(gf, out);
    printf("[encoder] number of nodes: %d\n", ggml_graph_n_nodes(gf));
    ggml_graph_compute_with_ctx(encoder_ctx, gf, 4 /* num_threads */);

    // copy output to main context
    struct ggml_tensor * encoded = ggml_new_tensor_2d(main_ctx, GGML_TYPE_F32, out->ne[0], out->ne[1]);
    memcpy(encoded->data, out->data, ggml_nbytes(out));
    ggml_set_name(encoded, "encoded");

    ggml_free(encoder_ctx);

    return encoded;
}