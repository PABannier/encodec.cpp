/* Implementation of an LSTM layer for Encodec.

An LSTM layer can be included in the Encodec model computational graph. Indeed, the LSTM
layer computational graph would grow with the sequence length, quickly exceeding the
limits imposed by ggml.

To circumvent this issue, this implementation builds a computational graph for a LSTM
cell forward pass. Once expanded, once only needs to change the inputs of the LSTM cell
(cell states, hidden states, inputs and weights) to perform the forward pass.

This implementation supports multiple LSTM layers.

Note that the final output tensor is stored in a main context (outside of the LSTM context)
which is freed after the computation is done.
*/

#include <cstring>
#include <vector>
#include <cassert>

#include "ggml.h"
#include "lstm.h"

static const int N_THREADS = 4;
static const int MB = 1024 * 1024;

// Number of nodes in the LSTM cell computational graph
static const int LSTM_CELL_NUM_NODES = 24;


struct encodec_lstm_cell {
    // Input cell tensor, [input_dim]
    struct ggml_tensor *x;

    // Cell state, [hidden_dim]
    struct ggml_tensor *c;
    // Hidden state, [hidden_dim]
    struct ggml_tensor *h;

    // [input_dim, hidden_dim * 4]
    struct ggml_tensor *weight_ih;
    // [hidden_dim, hidden_dim * 4]
    struct ggml_tensor *weight_hh;

    // [hidden_dim * 4]
    struct ggml_tensor *bias_ih;
    // [hidden_dim * 4]
    struct ggml_tensor *bias_hh;
};

struct encodec_lstm_cell_output {
    // Cell state, [hidden_dim]
    struct ggml_tensor *c;
    // Hidden state, [hidden_dim]
    struct ggml_tensor *h;
};

struct encodec_lstm_model {
    struct encodec_lstm_cell cell;

    // ggml context for main computation
    // This is the context in which the final output tensor is stored
    struct ggml_context *main_ctx;
    // ggml context for lstm computation
    struct ggml_context *lstm_ctx;

    // Input dimension
    int64_t input_dim;
    // Hidden dimension
    int64_t hidden_dim;
    // Number of timesteps
    int64_t n_timesteps;

    // Input tensor, [n_timesteps, input_dim] where innermost dimension is input_dim
    struct ggml_tensor *input;

    // Weight matrices
    encodec_lstm_layers layers;
};

static struct encodec_lstm_cell_output forward_pass_lstm_cell(struct encodec_lstm_model * lstm) {
    auto & ctx0 = lstm->lstm_ctx;

    const int hidden_dim = lstm->hidden_dim;

    auto & cell = lstm->cell;

    auto & x_t = cell.x;
    auto & c_t = cell.c;
    auto & h_t = cell.h;

    auto & weight_ih = cell.weight_ih;
    auto & weight_hh = cell.weight_hh;
    auto & bias_ih   = cell.bias_ih;
    auto & bias_hh   = cell.bias_hh;

    struct encodec_lstm_cell_output output;

    struct ggml_tensor *inp_gates = ggml_mul_mat(ctx0, weight_ih, x_t);
    inp_gates = ggml_add(ctx0, inp_gates, bias_ih);

    struct ggml_tensor *hid_gates = ggml_mul_mat(ctx0, weight_hh, h_t);
    hid_gates = ggml_add(ctx0, hid_gates, bias_hh);

    struct ggml_tensor *out_gates = ggml_add(ctx0, inp_gates, hid_gates);

    struct ggml_tensor *i_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 0 * sizeof(float) * hidden_dim));
    struct ggml_tensor *f_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 1 * sizeof(float) * hidden_dim));
    struct ggml_tensor *g_t = ggml_tanh   (ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 2 * sizeof(float) * hidden_dim));
    struct ggml_tensor *o_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 3 * sizeof(float) * hidden_dim));

    c_t = ggml_add(ctx0, ggml_mul(ctx0, f_t, c_t), ggml_mul(ctx0, i_t, g_t));
    h_t = ggml_mul(ctx0, o_t, ggml_tanh(ctx0, c_t));

    output.c = c_t;
    output.h = h_t;

    return output;
}

static struct ggml_tensor * forward_pass_lstm(struct encodec_lstm_model *lstm) {
    const int input_dim   = lstm->input_dim;
    const int hidden_dim  = lstm->hidden_dim;
    const int n_timesteps = lstm->n_timesteps;

    auto & main_ctx  = lstm->main_ctx;
    auto & lstm_ctx  = lstm->lstm_ctx;

    auto & cell = lstm->cell;

    struct ggml_tensor * final_output = ggml_new_tensor_2d(main_ctx, GGML_TYPE_F32, hidden_dim, n_timesteps);

    static size_t ctx_size = LSTM_CELL_NUM_NODES * ggml_tensor_overhead();
    ctx_size += ggml_graph_overhead();
    ctx_size += 256 * MB;  // Extra memory for safety

    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    lstm_ctx = ggml_init(params);

    struct ggml_cgraph * gf = ggml_new_graph(lstm_ctx);

    cell.x = ggml_new_tensor_1d(lstm_ctx, GGML_TYPE_F32, input_dim);
    cell.c = ggml_new_tensor_1d(lstm_ctx, GGML_TYPE_F32, hidden_dim);
    cell.h = ggml_new_tensor_1d(lstm_ctx, GGML_TYPE_F32, hidden_dim);

    ggml_set_input(cell.x);
    ggml_set_param(lstm_ctx, cell.c);
    ggml_set_param(lstm_ctx, cell.h);

    cell.weight_ih = ggml_new_tensor_2d(lstm_ctx, GGML_TYPE_F32,  input_dim, hidden_dim * 4);
    cell.weight_hh = ggml_new_tensor_2d(lstm_ctx, GGML_TYPE_F32, hidden_dim, hidden_dim * 4);

    ggml_set_param(lstm_ctx, cell.weight_ih);
    ggml_set_param(lstm_ctx, cell.weight_hh);

    cell.bias_ih = ggml_new_tensor_1d(lstm_ctx, GGML_TYPE_F32, hidden_dim * 4);
    cell.bias_hh = ggml_new_tensor_1d(lstm_ctx, GGML_TYPE_F32, hidden_dim * 4);

    ggml_set_param(lstm_ctx, cell.bias_ih);
    ggml_set_param(lstm_ctx, cell.bias_hh);

    struct encodec_lstm_cell_output output = forward_pass_lstm_cell(lstm);
    ggml_build_forward_expand(gf, output.h);

    printf("[lstm] number of nodes: %d\n", ggml_graph_n_nodes(gf));

    int n_layers = lstm->layers.size();

    for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx) {

        struct ggml_tensor * input_to_layer;
        if (layer_idx == 0) {
            input_to_layer = lstm->input;
        } else {
            input_to_layer = final_output;
        }

        auto & layer = lstm->layers[layer_idx];

        // Reset cell states
        ggml_set_zero(cell.c);
        ggml_set_zero(cell.h);

        // Copy layer weights and biases
        memcpy(cell.weight_ih->data, layer.weight_ih->data, ggml_nbytes(layer.weight_ih));
        memcpy(cell.weight_hh->data, layer.weight_hh->data, ggml_nbytes(layer.weight_hh));
        memcpy(cell.bias_ih->data  , layer.bias_ih->data  , ggml_nbytes(layer.bias_ih));
        memcpy(cell.bias_hh->data  , layer.bias_hh->data  , ggml_nbytes(layer.bias_hh));

        for (int timestep = 0; timestep < n_timesteps; ++timestep) {
            memcpy(
                cell.x->data,
                (float *) ((char *) input_to_layer->data + timestep * input_dim * sizeof(float)),
                input_dim * sizeof(float)
            );

            if (timestep > 0) {
                memcpy(cell.c->data, output.c->data, hidden_dim * sizeof(float));
                memcpy(cell.h->data, output.h->data, hidden_dim * sizeof(float));
            }

            ggml_graph_compute_with_ctx(lstm_ctx, gf, N_THREADS);

            float * dst_ptr = (float *) ((char *) ggml_get_data(final_output) + timestep * hidden_dim * sizeof(float));
            memcpy(dst_ptr, output.h->data, hidden_dim * sizeof(float));
        }
    }

    ggml_free(lstm_ctx);

    return final_output;
}

struct ggml_tensor * encodec_lstm(struct ggml_context * main_ctx,
                                  struct ggml_tensor  * inp,
                                  encodec_lstm_layers   layers) {
    encodec_lstm_model lstm;

    lstm.main_ctx = main_ctx;

    lstm.n_timesteps = inp->ne[0];
    lstm.input_dim   = inp->ne[1];
    lstm.hidden_dim  = layers[0].weight_ih->ne[1] / 4;

    // For simplicity purposes, we only support input_dim == hidden_dim as it is the only
    // case for Encodec.
    assert(lstm.input_dim == lstm.hidden_dim);

    // [n_timesteps, input_dim] -> [input_dim, n_timesteps]
    lstm.input = ggml_cont(main_ctx, ggml_transpose(main_ctx, inp));

    lstm.layers = layers;

    struct ggml_tensor * output = forward_pass_lstm(&lstm);

    // [hidden_dim, n_timesteps] -> [n_timesteps, hidden_dim]
    output = ggml_cont(main_ctx, ggml_transpose(main_ctx, output));

    return output;
}
