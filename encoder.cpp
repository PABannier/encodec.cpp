#include "encoder.h"
#include "ggml.h"
#include "util.h"

#include <cmath>

static int get_extra_padding_for_conv_1d(ggml_tensor * inp, float kernel_size, float stride, float padding_total) {
    float length = inp->ne[0];
    float n_frames = (length - kernel_size + padding_total) / stride + 1.0f;
    int ideal_length = (std::ceilf(n_frames) - 1) * stride + (kernel_size - padding_total);
    return ideal_length - length;
}

static struct ggml_tensor * pad_1d(ggml_context * ctx0, ggml_tensor * inp, int padding_left, int padding_right) {
    int length = inp->ne[0];
    int dim = inp->ne[1];
    ENCODEC_ASSERT(padding_left  >= 0);
    ENCODEC_ASSERT(padding_right >= 0);

    const int max_pad = std::max(padding_left, padding_right);
    int extra_pad = 0;

    if (length <= max_pad) {
        extra_pad = max_pad - length + 1;
        int padding[2] = {0, extra_pad};
        inp = ggml_pad_1d_constant(ctx0, inp, padding, 0);
    }

    int padding[2] = {padding_left, padding_right};
    struct ggml_tensor * padded = ggml_pad_1d_reflective(ctx0, inp, padding);

    const int end = padded->ne[0] - extra_pad;

    struct ggml_tensor *dest = ggml_view_2d(ctx0, padded, end, dim, padded->nb[1], 0);

    return dest;
}

static struct ggml_tensor * unpad_1d(ggml_context * ctx0, ggml_tensor * inp, int padding_left, int padding_right) {
    int length = inp->ne[0];
    int dim    = inp->ne[1];

    ENCODEC_ASSERT(padding_left  >= 0);
    ENCODEC_ASSERT(padding_right >= 0);
    ENCODEC_ASSERT(padding_left + padding_right <= length);

    int end = length - padding_right;

    int offset = padding_left * inp->nb[1];
    struct ggml_tensor * dst = ggml_view_2d(ctx0, inp, end, dim, inp->nb[1], offset);

    return dst;
}

struct ggml_tensor * strided_conv_1d(
            ggml_context * ctx0,
             ggml_tensor * inp,
             ggml_tensor * conv_w,
             ggml_tensor * conv_b,
                     int   stride) {
    int kernel_size   = conv_w->ne[0];
    int padding_total = kernel_size - stride;

    int extra_padding = get_extra_padding_for_conv_1d(inp, kernel_size, stride, padding_total);

    struct ggml_tensor * padded_inp = pad_1d(ctx0, inp, padding_total, extra_padding);

    struct ggml_tensor * dst = ggml_conv_1d(ctx0, conv_w, padded_inp, stride);

    // add bias
    dst = ggml_transpose(ctx0, dst);
    dst = ggml_add(ctx0, ggml_repeat(ctx0, conv_b, dst), dst);
    dst = ggml_cont(ctx0, ggml_transpose(ctx0, dst));

    return dst;
}

struct ggml_tensor * forward_pass_lstm_unilayer(
    struct ggml_context * ctx0,
    struct ggml_tensor * inp,
    struct ggml_tensor * weight_ih,
    struct ggml_tensor * weight_hh,
    struct ggml_tensor * bias_ih,
    struct ggml_tensor * bias_hh) {

    const int input_dim  = inp->ne[1];
    const int hidden_dim = weight_ih->ne[1]/4;
    const int seq_length = inp->ne[0];

    struct ggml_tensor * hs = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, seq_length);

    struct ggml_tensor * c_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_dim); 
    struct ggml_tensor * h_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_dim);

    ggml_set_zero(h_t);

    struct ggml_tensor * current = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    
    for (int t = 0; t < seq_length; t++) {
        struct ggml_tensor * x_t = ggml_view_1d(ctx0, current, input_dim, t*current->nb[1]);

        struct ggml_tensor * inp_gates = ggml_mul_mat(ctx0, weight_ih, x_t);
        inp_gates = ggml_add(ctx0, inp_gates, bias_ih);

        struct ggml_tensor * hid_gates = ggml_mul_mat(ctx0, weight_hh, h_t);
        hid_gates = ggml_add(ctx0, hid_gates, bias_hh);

        struct ggml_tensor * out_gates = ggml_add(ctx0, inp_gates, hid_gates);

        struct ggml_tensor * i_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 0*sizeof(float)*hidden_dim));
        struct ggml_tensor * f_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 1*sizeof(float)*hidden_dim));
        struct ggml_tensor * g_t = ggml_tanh   (ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 2*sizeof(float)*hidden_dim));
        struct ggml_tensor * o_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 3*sizeof(float)*hidden_dim));

        c_t = ggml_add(ctx0, ggml_mul(ctx0, f_t, c_t), ggml_mul(ctx0, i_t, g_t));
        h_t = ggml_mul(ctx0, o_t, ggml_tanh(ctx0, c_t));

        hs = ggml_set_1d(ctx0, hs, h_t, t*hs->nb[1]);
    }

    hs = ggml_cont(ctx0, ggml_transpose(ctx0, hs));

    return hs;
}

struct ggml_tensor * strided_conv_transpose_1d(
            ggml_context * ctx0,
             ggml_tensor * inp,
             ggml_tensor * conv_w,
             ggml_tensor * conv_b,
                     int   stride) {
    int kernel_size   = conv_w->ne[0];
    int padding_total = kernel_size - stride;

    struct ggml_tensor * dst = ggml_transpose_conv_1d(ctx0, conv_w, inp, stride);

    // add bias
    dst = ggml_transpose(ctx0, dst);
    dst = ggml_add(ctx0, ggml_repeat(ctx0, conv_b, dst), dst);
    dst = ggml_cont(ctx0, ggml_transpose(ctx0, dst));

    int padding_right = std::ceilf(padding_total);
    int padding_left = padding_total - padding_right;

    struct ggml_tensor * unpadded = unpad_1d(ctx0, dst, padding_left, padding_right);
    unpadded = ggml_cont(ctx0, unpadded);

    return unpadded;
}
