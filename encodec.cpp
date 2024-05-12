#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

#include "encodec.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define ENCODEC_FILE_MAGIC 'ggml'
#define ENCODEC_MAX_NODES 100000  // This is very high because of the LSTM layer growing with the sequence length

static const size_t MB = 1024 * 1024;

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

struct encodec_encoder {
    struct ggml_tensor *init_conv_w;
    struct ggml_tensor *init_conv_b;

    encodec_lstm lstm;

    struct ggml_tensor *final_conv_w;
    struct ggml_tensor *final_conv_b;

    std::vector<encodec_encoder_block> blocks;
};

struct encodec_quant_block {
    struct ggml_tensor *embed;
};

struct encodec_quantizer {
    std::vector<encodec_quant_block> blocks;
};

struct encodec_decoder_block {
    // upsampling layers
    struct ggml_tensor *us_conv_w;
    struct ggml_tensor *us_conv_b;

    // conv1
    struct ggml_tensor *conv_1_w;
    struct ggml_tensor *conv_1_b;

    // conv2
    struct ggml_tensor *conv_2_w;
    struct ggml_tensor *conv_2_b;

    // shortcut
    struct ggml_tensor *conv_sc_w;
    struct ggml_tensor *conv_sc_b;
};

struct encodec_decoder {
    struct ggml_tensor *init_conv_w;
    struct ggml_tensor *init_conv_b;

    encodec_lstm lstm;

    struct ggml_tensor *final_conv_w;
    struct ggml_tensor *final_conv_b;

    std::vector<encodec_decoder_block> blocks;
};

struct encodec_model {
    encodec_hparams hparams;

    encodec_encoder encoder;
    encodec_quantizer quantizer;
    encodec_decoder decoder;

    // context
    struct ggml_context *ctx;
    int n_loaded;

    ggml_backend_t backend = NULL;

    ggml_backend_buffer_t buffer_w;

    std::map<std::string, struct ggml_tensor *> tensors;
};

struct encodec_context {
    encodec_model model;

    // buffer for model evaluation
    ggml_backend_buffer_t buf_compute;

    // custom allocrator
    ggml_gallocr_t allocr;

    // intermediate steps
    struct ggml_tensor *encoded = NULL;  // Encoded audio
    struct ggml_tensor *codes = NULL;    // Quantized representation of audio in codebook
    struct ggml_tensor *decoded = NULL;  // Reconstructed audio from codes

    std::vector<int32_t> out_codes;
    std::vector<float> out_audio;

    // statistics
    encodec_statistics stats;
};

typedef enum {
    // Run the end-to-end encoder-decoder pipeline
    full = 0,
    // Encode an audio (encoder + quantizer encode)
    encode = 1,
    // Decode an audio from a compressed representation (quantizer decode + decoder)
    decode = 2,
} encodec_run_mode;

template <typename T>
static void read_safe(std::ifstream &infile, T &dest) {
    infile.read((char *)&dest, sizeof(T));
}

static void ggml_log_callback_default(ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
}

static void encodec_sigmoid_impl(
    struct ggml_tensor *dst,
    const struct ggml_tensor *src,
    int ith,
    int nth,
    void *userdata) {
    GGML_ASSERT(userdata == NULL);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    const float *src_data = ggml_get_data_f32(src);
    float *dst_data = ggml_get_data_f32(dst);

    const int ne = (int)ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i) {
        dst_data[i] = 1.0f / (1.0f + expf(-src_data[i]));
    }
}

static struct ggml_tensor *encodec_sigmoid(
    struct ggml_context *ctx,
    struct ggml_tensor *x) {
    return ggml_map_custom1(ctx, x, encodec_sigmoid_impl, GGML_N_TASKS_MAX, NULL);
}

static int get_extra_padding_for_conv_1d(
    struct ggml_tensor *inp,
    float kernel_size,
    float stride,
    float padding_total) {
    float length = inp->ne[0];
    float n_frames = (length - kernel_size + padding_total) / stride + 1.0f;
    int ideal_length = (ceilf(n_frames) - 1) * stride + (kernel_size - padding_total);
    return ideal_length - length;
}

static struct ggml_tensor *pad_1d(
    struct ggml_context *ctx0,
    struct ggml_tensor *inp,
    int padding_left,
    int padding_right) {
    int length = inp->ne[0];
    int dim = inp->ne[1];

    const int max_pad = std::max(padding_left, padding_right);
    int extra_pad = 0;

    if (length <= max_pad) {
        extra_pad = max_pad - length + 1;

        // constant padding
        struct ggml_tensor *out = ggml_new_tensor_2d(ctx0, inp->type, length + extra_pad, dim);
        ggml_set_zero(out);
        out = ggml_set_2d(ctx0, out, inp, out->nb[1], 0);
    }

    struct ggml_tensor *padded = ggml_pad_reflec_1d(ctx0, inp, padding_left, padding_right);

    const int end = padded->ne[0] - extra_pad;
    struct ggml_tensor *dest = ggml_view_2d(ctx0, padded, end, dim, padded->nb[1], 0);

    return dest;
}

static int32_t get_num_codebooks(float bandwidth, int hop_length, float sample_rate) {
    // The number of codebooks is determined by the bandwidth selected.
    // Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8),
    // 12 kbps (n_q = 16) and 24kbps (n_q = 32).
    return (int32_t)ceilf(1000 * bandwidth / (ceilf(sample_rate / hop_length) * 10));
}

static int32_t get_bandwidth_per_quantizer(int bins, float frame_rate) {
    return log2f((float)bins) * frame_rate;
}

static int32_t get_num_quantizers_for_bandwidth(int bins, float frame_rate, float bandwidth) {
    float bw_per_q = get_bandwidth_per_quantizer(bins, frame_rate);
    int32_t n_q = MAX(1, floorf(bandwidth * 1000 / bw_per_q));
    return n_q;
}

static struct ggml_tensor *unpad_1d(
    struct ggml_context *ctx0,
    struct ggml_tensor *inp,
    int padding_left,
    int padding_right) {
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

static struct ggml_tensor *strided_conv_1d(
    ggml_context *ctx0,
    ggml_tensor *inp,
    ggml_tensor *conv_w,
    ggml_tensor *conv_b,
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

static struct ggml_tensor *strided_conv_transpose_1d(
    struct ggml_context *ctx0,
    struct ggml_tensor *inp,
    struct ggml_tensor *conv_w,
    struct ggml_tensor *conv_b,
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

static struct ggml_tensor *forward_pass_lstm_unilayer(
    struct ggml_context *ctx0,
    ggml_gallocr_t *allocr,
    struct ggml_tensor *inp,
    struct ggml_tensor *weight_ih,
    struct ggml_tensor *weight_hh,
    struct ggml_tensor *bias_ih,
    struct ggml_tensor *bias_hh) {
    const int input_dim = inp->ne[1];
    const int hidden_dim = weight_ih->ne[1] / 4;
    const int seq_length = inp->ne[0];

    struct ggml_tensor *hs = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, seq_length);
    ggml_set_input(hs);

    struct ggml_tensor *c_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_dim);
    ggml_set_name(c_t, "lstm_c_t");

    struct ggml_tensor *h_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_dim);
    ggml_set_name(h_t, "lstm_h_t");

    struct ggml_tensor *current = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    for (int t = 0; t < seq_length; t++) {
        struct ggml_tensor *x_t = ggml_view_1d(ctx0, current, input_dim, t * current->nb[1]);

        struct ggml_tensor *inp_gates = ggml_mul_mat(ctx0, weight_ih, x_t);
        inp_gates = ggml_add(ctx0, inp_gates, bias_ih);

        struct ggml_tensor *hid_gates = ggml_mul_mat(ctx0, weight_hh, h_t);
        hid_gates = ggml_add(ctx0, hid_gates, bias_hh);

        struct ggml_tensor *out_gates = ggml_add(ctx0, inp_gates, hid_gates);

        struct ggml_tensor *i_t = encodec_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 0 * sizeof(float) * hidden_dim));
        struct ggml_tensor *f_t = encodec_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 1 * sizeof(float) * hidden_dim));
        struct ggml_tensor *g_t = ggml_tanh      (ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 2 * sizeof(float) * hidden_dim));
        struct ggml_tensor *o_t = encodec_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 3 * sizeof(float) * hidden_dim));

        c_t = ggml_add(ctx0, ggml_mul(ctx0, f_t, c_t), ggml_mul(ctx0, i_t, g_t));

        h_t = ggml_mul(ctx0, o_t, ggml_tanh(ctx0, c_t));

        hs = ggml_set_1d(ctx0, hs, h_t, t * hs->nb[1]);
    }

    hs = ggml_cont(ctx0, ggml_transpose(ctx0, hs));

    return hs;
}

bool encodec_load_model_weights(std::ifstream &infile, encodec_model &model, int n_gpu_layers) {
    // verify magic (i.e. ggml signature in hex format)
    {
        uint32_t magic;
        read_safe(infile, magic);
        if (magic != ENCODEC_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file (bad magic)\n", __func__);
            return false;
        }
    }

    // load hparams
    {
        auto &hparams = model.hparams;

        read_safe(infile, hparams.in_channels);
        read_safe(infile, hparams.hidden_dim);
        read_safe(infile, hparams.n_filters);
        read_safe(infile, hparams.kernel_size);
        read_safe(infile, hparams.residual_kernel_size);
        // read_safe(infile, hparams.ratios);
        read_safe(infile, hparams.n_bins);
        read_safe(infile, hparams.bandwidth);
        read_safe(infile, hparams.sr);
        read_safe(infile, hparams.ftype);

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        // printf("%s: in_channels = %d\n", __func__, hparams.in_channels);
        // printf("%s: hidden_dim  = %d\n", __func__, hparams.hidden_dim);
        // printf("%s: n_filters   = %d\n", __func__, hparams.n_filters);
        // printf("%s: kernel_size = %d\n", __func__, hparams.kernel_size);
        // printf("%s: res_kernel  = %d\n", __func__, hparams.residual_kernel_size);
        // // printf("%s: ratios      = %d\n", __func__, hparams.ratios);
        // printf("%s: n_bins      = %d\n", __func__, hparams.n_bins);
        // printf("%s: bandwidth   = %d\n", __func__, hparams.bandwidth);
        // printf("%s: sample_rate = %d\n", __func__, hparams.sr);
        // printf("%s: ftype       = %d\n", __func__, hparams.ftype);
        // printf("%s: qntvr       = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file (bad ftype value %d)\n",
                __func__, model.hparams.ftype);
        return 1;
    }

    auto &ctx = model.ctx;

    size_t buffer_size = 0;
    size_t n_tensors = 0;

    // Evaluating context size
    {
        const auto &hparams = model.hparams;

        const int in_channels = hparams.in_channels;
        const int hidden_dim = hparams.hidden_dim;
        const int n_filters = hparams.n_filters;
        const int kernel_size = hparams.kernel_size;
        const int res_kernel_sz = hparams.residual_kernel_size;
        const int n_bins = hparams.n_bins;
        const int *ratios = hparams.ratios;
        const int n_lstm_layers = hparams.n_lstm_layers;

        // encoder
        {
            int mult = 1;  // scaling factor for hidden size

            // initial conv1d layer
            buffer_size += in_channels * n_filters * kernel_size * ggml_type_size(wtype);  // weight
            buffer_size += n_filters * ggml_type_size(GGML_TYPE_F32);                      // bias

            // resnet blocks
            for (int i = 0; i < 4; i++) {
                // conv1
                buffer_size += res_kernel_sz * (mult * n_filters) * (mult * n_filters / 2) * ggml_type_size(wtype);  // weight
                buffer_size += (mult * n_filters / 2) * ggml_type_size(GGML_TYPE_F32);                               // bias

                // conv2
                buffer_size += (mult * n_filters / 2) * (mult * n_filters) * ggml_type_size(wtype);  // weight
                buffer_size += (mult * n_filters) * ggml_type_size(GGML_TYPE_F32);                   // bias

                // shortcut
                buffer_size += (mult * n_filters) * (mult * n_filters) * ggml_type_size(wtype);  // weight
                buffer_size += (mult * n_filters) * ggml_type_size(GGML_TYPE_F32);               // bias

                // downsampling layers
                buffer_size += (2 * ratios[3 - i]) * (mult * n_filters) * (mult * n_filters * 2) * ggml_type_size(wtype);  // weight
                buffer_size += (2 * mult * n_filters) * ggml_type_size(GGML_TYPE_F32);                                     // bias

                mult *= 2;
            }

            // lstm
            buffer_size += 2 * n_lstm_layers * (mult * n_filters) * (4 * mult * n_filters) * ggml_type_size(wtype);  // weight_ih and weight_hh
            buffer_size += 2 * n_lstm_layers * (4 * mult * n_filters) * ggml_type_size(GGML_TYPE_F32);               // bias_ih and bias_hh

            // final conv
            buffer_size += kernel_size * (mult * n_filters) * hidden_dim * ggml_type_size(wtype);  // weight
            buffer_size += hidden_dim * ggml_type_size(GGML_TYPE_F32);                             // bias
        }

        // decoder mirrors the encoder (same number of parameters), just double context size
        buffer_size *= 2;

        // quantizer
        int n_q = 32;                                                              // 32 is an upper bound on the number of codebooks.
        buffer_size += n_q * hidden_dim * n_bins * ggml_type_size(GGML_TYPE_F32);  // embed

        buffer_size += 10ull * MB;  // object overhead

        n_tensors = ((4 * 2) * 4 + 2 + 4 * n_lstm_layers + 2) * 2;  // encoder and decoder
        n_tensors += n_q * 1;                                       // quantizer

        printf("%s: ggml tensor size    = %d bytes\n", __func__, (int)sizeof(ggml_tensor));
        printf("%s: backend buffer size = %6.2f MB\n", __func__, buffer_size / (1024.0 * 1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * n_tensors,
            /* .mem_buffer = */ NULL,
            /* .no_alloc   = */ true,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > 0) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if (!model.backend) {
        // fallback to CPU backend
        fprintf(stderr, "%s: using CPU backend\n", __func__);
        model.backend = ggml_backend_cpu_init();
    }

    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
        return false;
    }

    // allocate weights buffer
    model.buffer_w = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // prepare memory for the weights
    {
        const auto &hparams = model.hparams;

        const int in_channels = hparams.in_channels;
        const int hidden_dim = hparams.hidden_dim;
        const int n_filters = hparams.n_filters;
        const int kernel_size = hparams.kernel_size;
        const int res_kernel_sz = hparams.residual_kernel_size;
        const int n_q = hparams.n_q;
        const int *ratios = hparams.ratios;
        const int n_bins = hparams.n_bins;

        // encoder
        {
            model.encoder.blocks.resize(4);

            int mult = 1;  // scaling factor for hidden size

            model.encoder.init_conv_w = ggml_new_tensor_3d(ctx, wtype, kernel_size, in_channels, mult * n_filters);
            model.encoder.init_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult * n_filters);

            model.tensors["encoder.model.0.conv.conv.weight"] = model.encoder.init_conv_w;
            model.tensors["encoder.model.0.conv.conv.bias"] = model.encoder.init_conv_b;

            for (int i = 0; i < 4; i++) {
                // conv1
                model.encoder.blocks[i].conv_1_w = ggml_new_tensor_3d(ctx, wtype, res_kernel_sz, mult * n_filters, mult * n_filters / 2);
                model.encoder.blocks[i].conv_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult * n_filters / 2);

                model.tensors["encoder.model." + std::to_string(3 * i + 1) + ".block.1.conv.conv.weight"] = model.encoder.blocks[i].conv_1_w;
                model.tensors["encoder.model." + std::to_string(3 * i + 1) + ".block.1.conv.conv.bias"] = model.encoder.blocks[i].conv_1_b;

                // conv2
                model.encoder.blocks[i].conv_2_w = ggml_new_tensor_3d(ctx, wtype, 1, mult * n_filters / 2, mult * n_filters);
                model.encoder.blocks[i].conv_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult * n_filters);

                model.tensors["encoder.model." + std::to_string(3 * i + 1) + ".block.3.conv.conv.weight"] = model.encoder.blocks[i].conv_2_w;
                model.tensors["encoder.model." + std::to_string(3 * i + 1) + ".block.3.conv.conv.bias"] = model.encoder.blocks[i].conv_2_b;

                // shortcut conv
                model.encoder.blocks[i].conv_sc_w = ggml_new_tensor_3d(ctx, wtype, 1, mult * n_filters, mult * n_filters);
                model.encoder.blocks[i].conv_sc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult * n_filters);

                model.tensors["encoder.model." + std::to_string(3 * i + 1) + ".shortcut.conv.conv.weight"] = model.encoder.blocks[i].conv_sc_w;
                model.tensors["encoder.model." + std::to_string(3 * i + 1) + ".shortcut.conv.conv.bias"] = model.encoder.blocks[i].conv_sc_b;

                // downsampling
                model.encoder.blocks[i].ds_conv_w = ggml_new_tensor_3d(ctx, wtype, 2 * ratios[3 - i], mult * n_filters, mult * n_filters * 2);
                model.encoder.blocks[i].ds_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult * n_filters * 2);

                model.tensors["encoder.model." + std::to_string(3 * (i + 1)) + ".conv.conv.weight"] = model.encoder.blocks[i].ds_conv_w;
                model.tensors["encoder.model." + std::to_string(3 * (i + 1)) + ".conv.conv.bias"] = model.encoder.blocks[i].ds_conv_b;

                mult *= 2;
            }

            // LSTM
            model.encoder.lstm.l0_ih_w = ggml_new_tensor_2d(ctx, wtype, mult * n_filters, 4 * mult * n_filters);
            model.encoder.lstm.l1_ih_w = ggml_new_tensor_2d(ctx, wtype, mult * n_filters, 4 * mult * n_filters);

            model.tensors["encoder.model.13.lstm.weight_ih_l0"] = model.encoder.lstm.l0_ih_w;
            model.tensors["encoder.model.13.lstm.weight_ih_l1"] = model.encoder.lstm.l1_ih_w;

            model.encoder.lstm.l0_hh_w = ggml_new_tensor_2d(ctx, wtype, mult * n_filters, 4 * mult * n_filters);
            model.encoder.lstm.l1_hh_w = ggml_new_tensor_2d(ctx, wtype, mult * n_filters, 4 * mult * n_filters);

            model.tensors["encoder.model.13.lstm.weight_hh_l0"] = model.encoder.lstm.l0_hh_w;
            model.tensors["encoder.model.13.lstm.weight_hh_l1"] = model.encoder.lstm.l1_hh_w;

            model.encoder.lstm.l0_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * mult * n_filters);
            model.encoder.lstm.l1_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * mult * n_filters);

            model.tensors["encoder.model.13.lstm.bias_ih_l0"] = model.encoder.lstm.l0_ih_b;
            model.tensors["encoder.model.13.lstm.bias_ih_l1"] = model.encoder.lstm.l1_ih_b;

            model.encoder.lstm.l0_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * mult * n_filters);
            model.encoder.lstm.l1_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * mult * n_filters);

            model.tensors["encoder.model.13.lstm.bias_hh_l0"] = model.encoder.lstm.l0_hh_b;
            model.tensors["encoder.model.13.lstm.bias_hh_l1"] = model.encoder.lstm.l1_hh_b;

            // final conv
            model.encoder.final_conv_w = ggml_new_tensor_3d(ctx, wtype, kernel_size, mult * n_filters, hidden_dim);
            model.encoder.final_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);

            model.tensors["encoder.model.15.conv.conv.weight"] = model.encoder.final_conv_w;
            model.tensors["encoder.model.15.conv.conv.bias"] = model.encoder.final_conv_b;
        }

        // decoder
        {
            model.decoder.blocks.resize(4);

            int mult = 16;  // 2**len(ratios)

            model.decoder.init_conv_w = ggml_new_tensor_3d(ctx, wtype, kernel_size, hidden_dim, mult * n_filters);
            model.decoder.init_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult * n_filters);

            model.tensors["decoder.model.0.conv.conv.weight"] = model.decoder.init_conv_w;
            model.tensors["decoder.model.0.conv.conv.bias"] = model.decoder.init_conv_b;

            // LSTM
            model.decoder.lstm.l0_ih_w = ggml_new_tensor_2d(ctx, wtype, mult * n_filters, 4 * mult * n_filters);
            model.decoder.lstm.l1_ih_w = ggml_new_tensor_2d(ctx, wtype, mult * n_filters, 4 * mult * n_filters);

            model.tensors["decoder.model.1.lstm.weight_ih_l0"] = model.decoder.lstm.l0_ih_w;
            model.tensors["decoder.model.1.lstm.weight_ih_l1"] = model.decoder.lstm.l1_ih_w;

            model.decoder.lstm.l0_hh_w = ggml_new_tensor_2d(ctx, wtype, mult * n_filters, 4 * mult * n_filters);
            model.decoder.lstm.l1_hh_w = ggml_new_tensor_2d(ctx, wtype, mult * n_filters, 4 * mult * n_filters);

            model.tensors["decoder.model.1.lstm.weight_hh_l0"] = model.decoder.lstm.l0_hh_w;
            model.tensors["decoder.model.1.lstm.weight_hh_l1"] = model.decoder.lstm.l1_hh_w;

            model.decoder.lstm.l0_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * mult * n_filters);
            model.decoder.lstm.l1_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * mult * n_filters);

            model.tensors["decoder.model.1.lstm.bias_ih_l0"] = model.decoder.lstm.l0_ih_b;
            model.tensors["decoder.model.1.lstm.bias_ih_l1"] = model.decoder.lstm.l1_ih_b;

            model.decoder.lstm.l0_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * mult * n_filters);
            model.decoder.lstm.l1_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * mult * n_filters);

            model.tensors["decoder.model.1.lstm.bias_hh_l0"] = model.decoder.lstm.l0_hh_b;
            model.tensors["decoder.model.1.lstm.bias_hh_l1"] = model.decoder.lstm.l1_hh_b;

            for (int i = 0; i < 4; i++) {
                // upsampling
                model.decoder.blocks[i].us_conv_w = ggml_new_tensor_3d(ctx, wtype, ratios[i] * 2, mult * n_filters / 2, mult * n_filters);
                model.decoder.blocks[i].us_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult * n_filters / 2);

                model.tensors["decoder.model." + std::to_string(3 * (i + 1)) + ".convtr.convtr.weight"] = model.decoder.blocks[i].us_conv_w;
                model.tensors["decoder.model." + std::to_string(3 * (i + 1)) + ".convtr.convtr.bias"] = model.decoder.blocks[i].us_conv_b;

                // conv1
                model.decoder.blocks[i].conv_1_w = ggml_new_tensor_3d(ctx, wtype, res_kernel_sz, mult * n_filters / 2, mult * n_filters / 4);
                model.decoder.blocks[i].conv_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult * n_filters / 4);

                model.tensors["decoder.model." + std::to_string(3 * (i + 1) + 1) + ".block.1.conv.conv.weight"] = model.decoder.blocks[i].conv_1_w;
                model.tensors["decoder.model." + std::to_string(3 * (i + 1) + 1) + ".block.1.conv.conv.bias"] = model.decoder.blocks[i].conv_1_b;

                // conv2
                model.decoder.blocks[i].conv_2_w = ggml_new_tensor_3d(ctx, wtype, 1, mult * n_filters / 4, mult * n_filters / 2);
                model.decoder.blocks[i].conv_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult * n_filters / 2);

                model.tensors["decoder.model." + std::to_string(3 * (i + 1) + 1) + ".block.3.conv.conv.weight"] = model.decoder.blocks[i].conv_2_w;
                model.tensors["decoder.model." + std::to_string(3 * (i + 1) + 1) + ".block.3.conv.conv.bias"] = model.decoder.blocks[i].conv_2_b;

                // shortcut
                model.decoder.blocks[i].conv_sc_w = ggml_new_tensor_3d(ctx, wtype, 1, mult * n_filters / 2, mult * n_filters / 2);
                model.decoder.blocks[i].conv_sc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult * n_filters / 2);

                model.tensors["decoder.model." + std::to_string(3 * (i + 1) + 1) + ".shortcut.conv.conv.weight"] = model.decoder.blocks[i].conv_sc_w;
                model.tensors["decoder.model." + std::to_string(3 * (i + 1) + 1) + ".shortcut.conv.conv.bias"] = model.decoder.blocks[i].conv_sc_b;

                mult /= 2;
            }

            model.decoder.final_conv_w = ggml_new_tensor_3d(ctx, wtype, kernel_size, n_filters, in_channels);
            model.decoder.final_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

            model.tensors["decoder.model.15.conv.conv.weight"] = model.decoder.final_conv_w;
            model.tensors["decoder.model.15.conv.conv.bias"] = model.decoder.final_conv_b;
        }

        // quantizer
        {
            model.quantizer.blocks.resize(n_q);

            for (int i = 0; i < n_q; i++) {
                model.quantizer.blocks[i].embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, n_bins);

                model.tensors["quantizer.vq.layers." + std::to_string(i) + "._codebook.embed"] = model.quantizer.blocks[i].embed;
            }
        }
    }

    // load weights
    {
        size_t total_size = 0;
        model.n_loaded = 0;

        std::vector<char> read_buf;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            read_safe(infile, n_dims);
            read_safe(infile, length);
            read_safe(infile, ftype);

            if (infile.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[3] = {1, 1, 1};
            for (int i = 0; i < n_dims; i++) {
                read_safe(infile, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> buf(length);
            infile.read(&buf[0], buf.size());
            name.assign(&buf[0], buf.size());

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            ggml_set_name(tensor, name.c_str());
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld, %lld], expected [%d, %d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], tensor->ne[2], ne[0], ne[1], ne[2]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ftype));

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
                return false;
            }

            if (ggml_backend_buffer_is_host(model.buffer_w)) {
                // for some backends such as CPU and Metal, the tensor data is in system memory and we can read directly into it
                infile.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(ggml_nbytes(tensor));
                infile.read(read_buf.data(), ggml_nbytes(tensor));
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            // printf("%48s - [%5d, %5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ne[2], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);

            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        printf("%s: model size = %8.2f MB\n", __func__, total_size / 1024.0 / 1024.0);
    }

    infile.close();

    return true;
}

struct ggml_tensor *encodec_forward_encoder(
    struct encodec_context *ectx,
    struct ggml_context *ctx0,
    struct ggml_tensor *inp) {
    if (!inp) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    const auto &model = ectx->model;
    const auto &hparams = model.hparams;
    const auto allocr = &ectx->allocr;

    const int *ratios = hparams.ratios;
    const int kernel_size = hparams.kernel_size;
    const int res_kernel_sz = hparams.residual_kernel_size;
    const int stride = hparams.stride;

    struct ggml_tensor *inpL = strided_conv_1d(
        ctx0, inp, model.encoder.init_conv_w, model.encoder.init_conv_b, stride);

    for (int layer_ix = 0; layer_ix < 4; layer_ix++) {
        encodec_encoder_block block = model.encoder.blocks[layer_ix];

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

        const encodec_lstm lstm = model.encoder.lstm;

        // first lstm layer
        struct ggml_tensor *hs1 = forward_pass_lstm_unilayer(
            ctx0, allocr, cur, lstm.l0_ih_w, lstm.l0_hh_w,
            lstm.l0_ih_b, lstm.l0_hh_b);

        // second lstm layer
        struct ggml_tensor *out = forward_pass_lstm_unilayer(
            ctx0, allocr, hs1, lstm.l1_ih_w, lstm.l1_hh_w,
            lstm.l1_ih_b, lstm.l1_hh_b);

        inpL = ggml_add(ctx0, inpL, out);
    }

    // final conv
    inpL = ggml_elu(ctx0, inpL);

    struct ggml_tensor *encoded_inp = strided_conv_1d(
        ctx0, inpL, model.encoder.final_conv_w, model.encoder.final_conv_b, stride);

    return encoded_inp;
}

struct ggml_tensor *encodec_forward_quantizer_encode(
    struct encodec_context *ectx,
    struct ggml_context *ctx0,
    struct ggml_tensor *encoded_inp) {
    if (!encoded_inp) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    const auto &model = ectx->model;
    const auto &hparams = model.hparams;
    const auto &allocr = ectx->allocr;

    const int n_bins = hparams.n_bins;
    const int sr = hparams.sr;
    const int bandwidth = hparams.bandwidth;
    const int hop_length = hparams.hop_length;

    const int frame_rate = (int)ceilf(sr / hop_length);
    const int n_q = get_num_quantizers_for_bandwidth(n_bins, frame_rate, bandwidth);

    const int seq_length = encoded_inp->ne[0];

    struct ggml_tensor *codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, seq_length, n_q);
    ggml_set_input(codes);

    struct ggml_tensor *inpL = ggml_cont(ctx0, ggml_transpose(ctx0, encoded_inp));
    struct ggml_tensor *residual = inpL;
    struct ggml_tensor *indices;

    for (int i = 0; i < n_q; i++) {
        encodec_quant_block block = model.quantizer.blocks[i];

        // compute distance
        // [seq_length, n_bins]
        struct ggml_tensor *dp = ggml_scale(
            ctx0, ggml_mul_mat(ctx0, block.embed, residual), -2.0f);

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
    struct encodec_context *ectx,
    struct ggml_context *ctx0,
    struct ggml_tensor *codes) {
    if (!codes) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    const auto &model = ectx->model;
    const auto &hparams = model.hparams;
    const auto &allocr = ectx->allocr;

    const int hidden_dim = hparams.hidden_dim;
    const int seq_length = codes->ne[0];

    const int n_bins = hparams.n_bins;
    const int sr = hparams.sr;
    const int bandwidth = hparams.bandwidth;
    const int hop_length = hparams.hop_length;

    const int frame_rate = (int)ceilf(sr / hop_length);
    const int n_q = get_num_quantizers_for_bandwidth(n_bins, frame_rate, bandwidth);

    assert(n_q == codes->ne[1]);

    struct ggml_tensor *quantized_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, seq_length);
    ggml_set_input(quantized_out);
    ggml_set_name(quantized_out, "quantized_out");

    for (int i = 0; i < n_q; i++) {
        encodec_quant_block block = model.quantizer.blocks[i];

        struct ggml_tensor *indices = ggml_view_1d(ctx0, codes, seq_length, i * codes->nb[1]);
        struct ggml_tensor *quantized = ggml_get_rows(ctx0, block.embed, indices);

        quantized_out = ggml_add(ctx0, quantized_out, quantized);
    }

    quantized_out = ggml_cont(ctx0, ggml_transpose(ctx0, quantized_out));

    return quantized_out;
}

struct ggml_tensor *encodec_forward_decoder(
    struct encodec_context *ectx,
    struct ggml_context *ctx0,
    struct ggml_tensor *quantized_out) {
    if (!quantized_out) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    const auto &model = ectx->model;
    const auto &hparams = model.hparams;
    const auto allocr = &ectx->allocr;

    const int *ratios = hparams.ratios;
    const int kernel_size = hparams.kernel_size;
    const int res_kernel_sz = hparams.residual_kernel_size;
    const int stride = hparams.stride;

    struct ggml_tensor *inpL = strided_conv_1d(
        ctx0, quantized_out, model.decoder.init_conv_w,
        model.decoder.init_conv_b, stride);

    // lstm
    {
        struct ggml_tensor *cur = inpL;

        const encodec_lstm lstm = model.decoder.lstm;

        // first lstm layer
        struct ggml_tensor *hs1 = forward_pass_lstm_unilayer(
            ctx0, allocr, cur, lstm.l0_ih_w, lstm.l0_hh_w,
            lstm.l0_ih_b, lstm.l0_hh_b);

        // second lstm layer
        struct ggml_tensor *out = forward_pass_lstm_unilayer(
            ctx0, allocr, hs1, lstm.l1_ih_w, lstm.l1_hh_w,
            lstm.l1_ih_b, lstm.l1_hh_b);

        inpL = ggml_add(ctx0, inpL, out);
    }

    for (int layer_ix = 0; layer_ix < 4; layer_ix++) {
        encodec_decoder_block block = model.decoder.blocks[layer_ix];

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
        ctx0, inpL, model.decoder.final_conv_w,
        model.decoder.final_conv_b, stride);

    return decoded_inp;
}

struct ggml_cgraph *encodec_build_graph(
    struct encodec_context *ectx,
    const float * inp_audio,
    const int n_samples,
    const encodec_run_mode mode) {
    assert(mode == encodec_run_mode::full || mode == encodec_run_mode::encode);

    const auto &model = ectx->model;
    const auto &hparams = model.hparams;
    const auto &allocr = ectx->allocr;

    const int n_q = hparams.n_q;

    // since we are using ggml-alloc, this buffer only needs enough space to hold the
    // ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead() * ENCODEC_MAX_NODES + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/buf.data(),
        /*.no_alloc   =*/true,  // skip allocating as we use ggml_alloc to allocate exact memory requirements
    };

    struct ggml_context *ctx0 = ggml_init(ggml_params);

    struct ggml_cgraph *gf = ggml_new_graph(ctx0);

    struct ggml_tensor *inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_samples);
    ggml_set_name(inp, "inp");
    ggml_set_input(inp);

    struct ggml_tensor *encoded = encodec_forward_encoder(ectx, ctx0, inp);
    struct ggml_tensor *codes = encodec_forward_quantizer_encode(ectx, ctx0, encoded);
    struct ggml_tensor *quantized = encodec_forward_quantizer_decode(ectx, ctx0, codes);
    struct ggml_tensor *decoded = encodec_forward_decoder(ectx, ctx0, quantized);

    switch (mode) {
        case encodec_run_mode::full: {
            ggml_build_forward_expand(gf, decoded);
        } break;
        case encodec_run_mode::encode: {
            ggml_build_forward_expand(gf, codes);
        } break;
        case encodec_run_mode::decode: {
            return NULL;
        } break;
        default: {
            fprintf(stderr, "%s: unknown run mode\n", __func__);
            return NULL;
        } break;
    }

    ggml_free(ctx0);

    ectx->encoded = encoded;
    ectx->codes = codes;
    ectx->decoded = decoded;

    return gf;
}

struct ggml_cgraph *encodec_build_graph(
    struct encodec_context *ectx,
    const int32_t * codes,
    const int n_codes,
    const encodec_run_mode mode) {
    assert(mode == encodec_run_mode::decode);

    const auto &model = ectx->model;
    const auto &hparams = model.hparams;
    const auto &allocr = ectx->allocr;

    const int n_bins = hparams.n_bins;
    const int sr = hparams.sr;
    const int bandwidth = hparams.bandwidth;
    const int hop_length = hparams.hop_length;

    const int frame_rate = (int)ceilf(sr / hop_length);
    const int n_q = get_num_quantizers_for_bandwidth(n_bins, frame_rate, bandwidth);

    if (n_codes % n_q != 0) {
        fprintf(stderr, "%s: invalid number of codes\n", __func__);
        return NULL;
    }

    const int N = n_codes / n_q;

    // since we are using ggml-alloc, this buffer only needs enough space to hold the
    // ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead() * ENCODEC_MAX_NODES + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/buf.data(),
        /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(ggml_params);

    struct ggml_cgraph *gf = ggml_new_graph(ctx0);

    struct ggml_tensor *inp_codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, N, n_q);
    ggml_set_name(inp_codes, "inp_codes");
    ggml_set_input(inp_codes);

    struct ggml_tensor *quantized = encodec_forward_quantizer_decode(ectx, ctx0, inp_codes);
    struct ggml_tensor *decoded = encodec_forward_decoder(ectx, ctx0, quantized);

    switch (mode) {
        case encodec_run_mode::decode: {
            ggml_build_forward_expand(gf, decoded);
        } break;
        default: {
            fprintf(stderr, "%s: unknown run mode\n", __func__);
            return NULL;
        } break;
    }

    ggml_free(ctx0);

    ectx->codes = inp_codes;
    ectx->decoded = decoded;

    return gf;
}

bool encodec_eval_internal(
    struct encodec_context *ectx,
    const float * raw_audio,
    const int n_samples,
    const int n_threads,
    const encodec_run_mode mode) {
    auto &model = ectx->model;
    auto &allocr = ectx->allocr;

    struct ggml_cgraph *gf = encodec_build_graph(ectx, raw_audio, n_samples, mode);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    // set the graph input
    struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "inp");
    ggml_backend_tensor_set(inp, raw_audio, 0, n_samples * ggml_element_size(inp));

    struct ggml_tensor * c_t = ggml_graph_get_tensor(gf, "lstm_c_t");
    ggml_set_zero(c_t);

    struct ggml_tensor * h_t = ggml_graph_get_tensor(gf, "lstm_h_t");
    ggml_set_zero(h_t);

    struct ggml_tensor * quantized_out = ggml_graph_get_tensor(gf, "quantized_out");
    if (quantized_out) {
        ggml_set_zero(quantized_out);
    }

    // run the computation
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif
    ggml_backend_graph_compute(model.backend, gf);

    return true;
}

bool encodec_eval_internal(
    struct encodec_context *ectx,
    const int32_t * codes,
    const int n_codes,
    const int n_threads,
    const encodec_run_mode mode) {
    auto &model = ectx->model;
    auto &allocr = ectx->allocr;

    struct ggml_cgraph *gf = encodec_build_graph(ectx, codes, n_codes, mode);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    // set data for input tensor
    struct ggml_tensor * inp_codes = ggml_graph_get_tensor(gf, "inp_codes");
    ggml_backend_tensor_set(inp_codes, codes, 0, n_codes * ggml_element_size(inp_codes));

    struct ggml_tensor * c_t = ggml_graph_get_tensor(gf, "lstm_c_t");
    ggml_set_zero(c_t);

    struct ggml_tensor * h_t = ggml_graph_get_tensor(gf, "lstm_h_t");
    ggml_set_zero(h_t);

    struct ggml_tensor * quantized_out = ggml_graph_get_tensor(gf, "quantized_out");
    if (quantized_out) {
        ggml_set_zero(quantized_out);
    }

    // run the computation
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif
    ggml_backend_graph_compute(model.backend, gf);

    return true;
}

bool encodec_eval(
    struct encodec_context *ectx,
    const float *raw_audio,
    const int n_samples,
    const int n_threads,
    const encodec_run_mode mode) {
    const int64_t t_start_us = ggml_time_us();

    // allocate the compute buffer
    {
        ectx->allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

        // create the graph for memory usage estimation
        struct ggml_cgraph *gf = encodec_build_graph(ectx, raw_audio, n_samples, mode);

        ggml_gallocr_reserve(ectx->allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(ectx->allocr, 0);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n\n", __func__, mem_size / 1024.0 / 1024.0);
    }

    // encodec eval
    if (!encodec_eval_internal(ectx, raw_audio, n_samples, n_threads, mode)) {
        fprintf(stderr, "%s: failed to run encodec eval\n", __func__);
        return false;
    }

    ectx->stats.t_compute_us = ggml_time_us() - t_start_us;

    return true;
}

bool encodec_eval(
    struct encodec_context *ectx,
    const int32_t *codes,
    const int n_codes,
    const int n_threads,
    const encodec_run_mode mode) {
    const int64_t t_start_ms = ggml_time_us();

    // allocate the compute buffer
    {
        ectx->allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

        // create the graph for memory usage estimation
        struct ggml_cgraph *gf = encodec_build_graph(ectx, codes, n_codes, mode);

        ggml_gallocr_reserve(ectx->allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(ectx->allocr, 0);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n\n", __func__, mem_size / 1024.0 / 1024.0);
    }

    // encodec eval
    if (!encodec_eval_internal(ectx, codes, n_codes, n_threads, mode)) {
        fprintf(stderr, "%s: failed to run encodec eval\n", __func__);
        return false;
    }

    ectx->stats.t_compute_us = ggml_time_us() - t_start_ms;

    return true;
}

bool encodec_reconstruct_audio(
        struct encodec_context *ectx,
        const float *raw_audio,
        const int n_samples,
        int n_threads) {
    if (raw_audio == nullptr) {
        fprintf(stderr, "%s: null input audio\n", __func__);
        return false;
    }

    if (!encodec_eval(ectx, raw_audio, n_samples, n_threads, encodec_run_mode::full)) {
        fprintf(stderr, "%s: failed to run encodec eval\n", __func__);
        return false;
    }

    if (!ectx->decoded) {
        fprintf(stderr, "%s: null decoded tensor\n", __func__);
        return false;
    }

    struct ggml_tensor *decoded = ectx->decoded;

    auto &out_audio = ectx->out_audio;

    int out_length = decoded->ne[0];
    out_audio.resize(out_length);

    ggml_backend_tensor_get(decoded, out_audio.data(), 0, out_length * ggml_element_size(decoded));

    return true;
}

bool encodec_compress_audio(
    struct encodec_context *ectx,
    const float * raw_audio,
    const int n_samples,
    int n_threads) {
    if (!encodec_eval(ectx, raw_audio, n_samples, n_threads, encodec_run_mode::encode)) {
        fprintf(stderr, "%s: failed to run encodec eval\n", __func__);
        return false;
    }

    if (!ectx->codes) {
        fprintf(stderr, "%s: null codes tensor\n", __func__);
        return false;
    }

    struct ggml_tensor *codes = ectx->codes;

    auto &out_codes = ectx->out_codes;

    int out_length = codes->ne[0] * codes->ne[1];
    out_codes.resize(out_length);

    ggml_backend_tensor_get(codes, out_codes.data(), 0, out_length * ggml_element_size(codes));

    return true;
}

bool encodec_decompress_audio(
    struct encodec_context *ectx,
    const int32_t * codes,
    const int n_codes,
    int n_threads) {
    if (!encodec_eval(ectx, codes, n_codes, n_threads, encodec_run_mode::decode)) {
        fprintf(stderr, "%s: failed to run encodec eval\n", __func__);
        return false;
    }

    if (!ectx->decoded) {
        fprintf(stderr, "%s: null decoded tensor\n", __func__);
        return false;
    }

    struct ggml_tensor *decoded = ectx->decoded;

    auto &out_audio = ectx->out_audio;

    int out_length = decoded->ne[0];
    out_audio.resize(out_length);

    ggml_backend_tensor_get(decoded, out_audio.data(), 0, out_length * ggml_element_size(decoded));

    return true;
}

// The offset parameter is used to adapt to two scenarios:
// 1. If offset is 0, it is assumed the file only contains the Encodec weights, hence
//    the model is loaded from the beginning of the file.
// 2. If offset is gt 0, it is assumed the file contains the weights and then the Encodec
//    model, hence the model is loaded from the offset. This is the case for Bark.
// Note that we used to have an encodec_load_model taking a reference to a file stream
// but it was removed to comply the C-header requirements.
struct encodec_context *encodec_load_model(const char* model_path, const int offset, int n_gpu_layers) {
    int64_t t_start_load_us = ggml_time_us();

    auto infile = std::ifstream(model_path, std::ios::binary);
    if (!infile) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, model_path);
        return nullptr;
    }

    if (offset > 0) {
        infile.seekg(offset);
    }

    struct encodec_context *ectx = new encodec_context();

    ectx->model = encodec_model();
    if (!encodec_load_model_weights(infile, ectx->model, n_gpu_layers)) {
        fprintf(stderr, "%s: failed to load model weights from '%s'\n", __func__, model_path);
        return {};
    }

    // pre-compute the number of codebooks required
    int bandwidth = ectx->model.hparams.bandwidth;
    int sr = ectx->model.hparams.sr;

    int hop_length = 1;
    for (int i = 0; i < 4; i++) {
        hop_length *= ectx->model.hparams.ratios[i];
    }
    ectx->model.hparams.hop_length = hop_length;

    ectx->model.hparams.n_q = get_num_codebooks(bandwidth, hop_length, sr);
    fprintf(stderr, "%s: n_q = %d\n", __func__, ectx->model.hparams.n_q);

    ectx->stats.t_load_us = ggml_time_us() - t_start_load_us;

    return ectx;
}

void encodec_free(struct encodec_context *ectx) {
    if (!ectx) {
        return;
    }

    if (ectx->model.ctx) {
        ggml_free(ectx->model.ctx);
    }

    if (ectx->buf_compute) {
        ggml_backend_buffer_free(ectx->buf_compute);
    }

    ggml_backend_buffer_free(ectx->model.buffer_w);
    ggml_backend_free(ectx->model.backend);

    delete ectx;
}

void encodec_set_target_bandwidth(struct encodec_context *ectx, int bandwidth) {
    ectx->model.hparams.bandwidth = bandwidth;
}

void encodec_set_sample_rate(struct encodec_context *ectx, int sample_rate) {
    ectx->model.hparams.sr = sample_rate;
}

const struct encodec_statistics* encodec_get_statistics(struct encodec_context *ectx) {
    if (!ectx) {
        fprintf(stderr, "%s: null context\n", __func__);
        return nullptr;
    }
    return &ectx->stats;
}

void encodec_reset_statistics(struct encodec_context *ectx) {
    if (!ectx) {
        fprintf(stderr, "%s: null context\n", __func__);
        return;
    }
    memset(&ectx->stats, 0, sizeof(ectx->stats));
}

float * encodec_get_audio(struct encodec_context *ectx) {
    if (!ectx) {
        fprintf(stderr, "%s: null context\n", __func__);
        return nullptr;
    }
    return ectx->out_audio.data();
}

int encodec_get_audio_size(struct encodec_context *ectx) {
    if (!ectx) {
        fprintf(stderr, "%s: null context\n", __func__);
        return 0;
    }
    return ectx->out_audio.size();
}

int32_t * encodec_get_codes(struct encodec_context *ectx) {
    if (!ectx) {
        fprintf(stderr, "%s: null context\n", __func__);
        return nullptr;
    }
    return ectx->out_codes.data();
}

int encodec_get_codes_size(struct encodec_context *ectx) {
    if (!ectx) {
        fprintf(stderr, "%s: null context\n", __func__);
        return 0;
    }
    return ectx->out_codes.size();
}
