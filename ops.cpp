#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "ggml.h"

#include "ops.h"

static int get_extra_padding_for_conv_1d(struct ggml_tensor *inp, float kernel_size,
                                         float stride, float padding_total) {
    float length = inp->ne[0];
    float n_frames = (length - kernel_size + padding_total) / stride + 1.0f;
    int ideal_length = (ceilf(n_frames) - 1) * stride + (kernel_size - padding_total);
    return ideal_length - length;
}

struct ggml_tensor *pad_1d(struct ggml_context *ctx0, struct ggml_tensor *inp,
                           int padding_left, int padding_right) {
    int length = inp->ne[0];
    int dim = inp->ne[1];

    const int max_pad = std::max(padding_left, padding_right);
    int extra_pad = 0;

    if (length <= max_pad) {
        extra_pad = max_pad - length + 1;

        // constant padding
        struct ggml_tensor *out = ggml_new_tensor_2d(ctx0, inp->type, length + extra_pad, dim);
        out = ggml_set_2d(ctx0, out, inp, out->nb[1], 0);
    }

    struct ggml_tensor *padded = ggml_pad_reflect_1d(ctx0, inp, padding_left, padding_right);

    const int end = padded->ne[0] - extra_pad;
    struct ggml_tensor *dest = ggml_view_2d(ctx0, padded, end, dim, padded->nb[1], 0);

    return dest;
}

struct ggml_tensor *unpad_1d(struct ggml_context *ctx0, struct ggml_tensor *inp,
                             int padding_left, int padding_right) {
    int length = inp->ne[0];
    int dim = inp->ne[1];

    assert(padding_left >= 0);
    assert(padding_right >= 0);
    assert(padding_left + padding_right <= length);

    int end = length - padding_right;

    int offset = padding_left * inp->nb[1];
    struct ggml_tensor *dst = ggml_view_2d(ctx0, inp, end, dim, inp->nb[1], offset);

    return dst;
}

struct ggml_tensor *strided_conv_1d(struct ggml_context *ctx0, struct ggml_tensor *inp,
                                    struct ggml_tensor *conv_w, struct ggml_tensor *conv_b,
                                    int stride) {
    int kernel_size = conv_w->ne[0];
    int padding_total = kernel_size - stride;
    int extra_padding = get_extra_padding_for_conv_1d(inp, kernel_size, stride, padding_total);

    struct ggml_tensor *padded_inp = pad_1d(ctx0, inp, padding_total, extra_padding);
    struct ggml_tensor *dst = ggml_conv_1d(ctx0, conv_w, padded_inp, stride, 0, 1);

    // add bias
    dst = ggml_transpose(ctx0, dst);
    dst = ggml_add(ctx0, ggml_repeat(ctx0, conv_b, dst), dst);
    dst = ggml_cont(ctx0, ggml_transpose(ctx0, dst));

    return dst;
}

struct ggml_tensor *strided_conv_transpose_1d(struct ggml_context *ctx0, struct ggml_tensor *inp,
                                              struct ggml_tensor *conv_w, struct ggml_tensor *conv_b,
                                              int stride) {
    struct ggml_tensor *dst = ggml_conv_transpose_1d(
        ctx0, conv_w, inp, stride, 0 /* p0 */, 1 /* d0 */);

    // add bias
    dst = ggml_transpose(ctx0, dst);
    dst = ggml_add(ctx0, ggml_repeat(ctx0, conv_b, dst), dst);
    dst = ggml_cont(ctx0, ggml_transpose(ctx0, dst));

    int kernel_size = conv_w->ne[0];
    int padding_total = kernel_size - stride;

    int padding_right = ceilf(padding_total);
    int padding_left = padding_total - padding_right;

    struct ggml_tensor *unpadded = unpad_1d(ctx0, dst, padding_left, padding_right);
    unpadded = ggml_cont(ctx0, unpadded);

    return unpadded;
}
