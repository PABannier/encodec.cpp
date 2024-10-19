#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#include "ops.h"

struct encodec_lstm {
    struct ggml_tensor *l0_ih_w;
    struct ggml_tensor *l0_hh_w;

    struct ggml_tensor *l0_ih_b;
    struct ggml_tensor *l0_hh_b;

    struct ggml_tensor *l1_ih_w;
    struct ggml_tensor *l1_hh_w;

    struct ggml_tensor *l1_ih_b;
    struct ggml_tensor *l1_hh_b;
};

struct ggml_tensor *forward_pass_lstm_unilayer(struct ggml_context *ctx0,
                                               struct ggml_tensor  *inp,
                                               struct ggml_tensor  *weight_ih,
                                               struct ggml_tensor  *weight_hh,
                                               struct ggml_tensor  *bias_ih,
                                               struct ggml_tensor  *bias_hh,
                                               char                *prefix) {
    const int seq_length = inp->ne[0];
    const int input_dim  = inp->ne[1];
    const int hidden_dim = weight_ih->ne[1] / 4;

    char ct_name[10];
    char ht_name[10];

    snprintf(ct_name, 10, "%s_ct", prefix);
    snprintf(ht_name, 10, "%s_ht", prefix);

    struct ggml_tensor *hs = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, seq_length);
    ggml_set_input(hs);

    struct ggml_tensor *c_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_dim);
    ggml_set_input(c_t);
    ggml_set_name(c_t, ct_name);

    struct ggml_tensor *h_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_dim);
    ggml_set_input(h_t);
    ggml_set_name(h_t, ht_name);

    struct ggml_tensor *current = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    for (int t = 0; t < seq_length; t++) {
        struct ggml_tensor *x_t = ggml_view_1d(ctx0, current, input_dim, t * current->nb[1]);

        struct ggml_tensor *inp_gates = ggml_mul_mat(ctx0, weight_ih, x_t);
        inp_gates = ggml_add(ctx0, inp_gates, bias_ih);

        struct ggml_tensor *hid_gates = ggml_mul_mat(ctx0, weight_hh, h_t);
        hid_gates = ggml_add(ctx0, hid_gates, bias_hh);

        struct ggml_tensor *out_gates = ggml_add(ctx0, inp_gates, hid_gates);

        struct ggml_tensor *i_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 0 * sizeof(float) * hidden_dim));
        struct ggml_tensor *f_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 1 * sizeof(float) * hidden_dim));
        struct ggml_tensor *g_t = ggml_tanh(ctx0   , ggml_view_1d(ctx0, out_gates, hidden_dim, 2 * sizeof(float) * hidden_dim));
        struct ggml_tensor *o_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 3 * sizeof(float) * hidden_dim));

        c_t = ggml_add(ctx0, ggml_mul(ctx0, f_t, c_t), ggml_mul(ctx0, i_t, g_t));

        h_t = ggml_mul(ctx0, o_t, ggml_tanh(ctx0, c_t));

        hs = ggml_set_1d(ctx0, hs, h_t, t * hs->nb[1]);
    }

    hs = ggml_cont(ctx0, ggml_transpose(ctx0, hs));

    return hs;
}
