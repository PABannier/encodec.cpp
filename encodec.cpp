#include <cmath>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "encodec.h"
#include "ggml.h"


static const size_t TENSOR_ALIGNMENT = 32;

// res + downsample block at some ratio
struct encodec_encoder_block {
    // conv1
    struct ggml_tensor * conv_1_w;
    struct ggml_tensor * conv_1_b;

    // conv2
    struct ggml_tensor * conv_2_w;
    struct ggml_tensor * conv_2_b;

    // shortcut
    struct ggml_tensor * conv_sc_w;
    struct ggml_tensor * conv_sc_b;

    // downsampling layers
    struct ggml_tensor * ds_conv_w;  
    struct ggml_tensor * ds_conv_b;
};

struct encodec_lstm {
    struct ggml_tensor * l0_ih_w;
    struct ggml_tensor * l0_hh_w;
    
    struct ggml_tensor * l0_ih_b;
    struct ggml_tensor * l0_hh_b;

    struct ggml_tensor * l1_ih_w;
    struct ggml_tensor * l1_hh_w;

    struct ggml_tensor * l1_ih_b;
    struct ggml_tensor * l1_hh_b;
};

struct encodec_encoder {
    struct ggml_tensor * init_conv_w;
    struct ggml_tensor * init_conv_b;

    encodec_lstm lstm;

    struct ggml_tensor * final_conv_w;
    struct ggml_tensor * final_conv_b;

    std::vector<encodec_encoder_block> blocks;
};

struct encodec_quant_block {
    struct ggml_tensor * inited;
    struct ggml_tensor * cluster_size;
    struct ggml_tensor * embed;
    struct ggml_tensor * embed_avg;
};

struct encodec_quantizer {
    std::vector<encodec_quant_block> blocks;
};

struct encodec_decoder_block {
    //upsampling layers
    struct ggml_tensor * us_conv_w;
    struct ggml_tensor * us_conv_b;

    // conv1
    struct ggml_tensor * conv_1_w;
    struct ggml_tensor * conv_1_b;

    // conv2
    struct ggml_tensor * conv_2_w;
    struct ggml_tensor * conv_2_b;

    // shortcut
    struct ggml_tensor * conv_sc_w;
    struct ggml_tensor * conv_sc_b;
};

struct encodec_decoder {
    struct ggml_tensor * init_conv_w;
    struct ggml_tensor * init_conv_b;

    encodec_lstm lstm;

    struct ggml_tensor * final_conv_w;
    struct ggml_tensor * final_conv_b;

    std::vector<encodec_decoder_block> blocks;
};

struct encodec_model {
    encodec_hparams hparams;

    encodec_encoder   encoder;
    encodec_quantizer quantizer;
    encodec_decoder   decoder;

    // context
    struct ggml_context * ctx;
    int n_loaded;

    std::map<std::string, struct ggml_tensor *> tensors;
};

template<typename T>
static void read_safe(std::ifstream& infile, T& dest) {
    infile.read((char*)& dest, sizeof(T));
}

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

static void ggml_disconnect_node_from_graph(ggml_tensor * t) {
    t->op = GGML_OP_NONE;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        t->src[i] = NULL;
    }
}

static void encodec_sigmoid_impl(struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    GGML_ASSERT(userdata == NULL);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    const float * src_data = ggml_get_data_f32(src);
    float * dst_data = ggml_get_data_f32(dst);

    const int ne = (int)ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i) {
        dst_data[i] = 1.0f / (1.0f + expf(-src_data[i]));
    }
}

static struct ggml_tensor * encodec_sigmoid(ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_custom1(ctx, x, encodec_sigmoid_impl, GGML_N_TASKS_MAX, NULL);
}

static int get_extra_padding_for_conv_1d(ggml_tensor * inp, float kernel_size, float stride, float padding_total) {
    float length = inp->ne[0];
    float n_frames = (length - kernel_size + padding_total) / stride + 1.0f;
    int ideal_length = (ceilf(n_frames) - 1) * stride + (kernel_size - padding_total);
    return ideal_length - length;
}

static struct ggml_tensor * pad_1d(ggml_context * ctx0, ggml_tensor * inp, int padding_left, int padding_right) {
    int length = inp->ne[0];
    int dim = inp->ne[1];

    const int max_pad = std::max(padding_left, padding_right);
    int extra_pad = 0;

    if (length <= max_pad) {
        extra_pad = max_pad - length + 1;

        // constant padding
        struct ggml_tensor * out = ggml_new_tensor_2d(ctx0, inp->type, length+extra_pad, dim);
        ggml_set_zero(out);
        out = ggml_set_2d(ctx0, out, inp, out->nb[1], 0);
    }

    struct ggml_tensor * padded = ggml_pad_reflec_1d(ctx0, inp, padding_left, padding_right);

    const int end = padded->ne[0] - extra_pad;
    struct ggml_tensor *dest = ggml_view_2d(ctx0, padded, end, dim, padded->nb[1], 0);

    return dest;
}

static struct ggml_tensor * unpad_1d(ggml_context * ctx0, ggml_tensor * inp, int padding_left, int padding_right) {
    int length = inp->ne[0];
    int dim    = inp->ne[1];

    assert(padding_left  >= 0);
    assert(padding_right >= 0);
    assert(padding_left + padding_right <= length);

    int end = length - padding_right;

    int offset = padding_left * inp->nb[1];
    struct ggml_tensor * dst = ggml_view_2d(ctx0, inp, end, dim, inp->nb[1], offset);

    return dst;
}

static struct ggml_tensor * strided_conv_1d(
            ggml_context * ctx0,
             ggml_tensor * inp,
             ggml_tensor * conv_w,
             ggml_tensor * conv_b,
                     int   stride) {
    int kernel_size   = conv_w->ne[0];
    int padding_total = kernel_size - stride;
    int extra_padding = get_extra_padding_for_conv_1d(inp, kernel_size, stride, padding_total);

    struct ggml_tensor * padded_inp = pad_1d(ctx0, inp, padding_total, extra_padding);
    struct ggml_tensor * dst = ggml_conv_1d(ctx0, conv_w, padded_inp, stride, 0, 1);

    // add bias
    dst = ggml_transpose(ctx0, dst);
    dst = ggml_add(ctx0, ggml_repeat(ctx0, conv_b, dst), dst);
    dst = ggml_cont(ctx0, ggml_transpose(ctx0, dst));

    return dst;
}

static struct ggml_tensor * forward_pass_lstm_unilayer(
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

    h_t = ggml_set_zero(h_t);
    c_t = ggml_set_zero(c_t);

    struct ggml_tensor * current = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    for (int t = 0; t < seq_length; t++) {
        struct ggml_tensor * x_t = ggml_view_1d(ctx0, current, input_dim, t*current->nb[1]);

        struct ggml_tensor * inp_gates = ggml_mul_mat(ctx0, weight_ih, x_t);
        inp_gates = ggml_add(ctx0, inp_gates, bias_ih);

        struct ggml_tensor * hid_gates = ggml_mul_mat(ctx0, weight_hh, h_t);
        hid_gates = ggml_add(ctx0, hid_gates, bias_hh);

        struct ggml_tensor * out_gates = ggml_add(ctx0, inp_gates, hid_gates);

        struct ggml_tensor * i_t = encodec_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 0*sizeof(float)*hidden_dim));
        struct ggml_tensor * f_t = encodec_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 1*sizeof(float)*hidden_dim));
        struct ggml_tensor * g_t = ggml_tanh   (ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 2*sizeof(float)*hidden_dim));
        struct ggml_tensor * o_t = encodec_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 3*sizeof(float)*hidden_dim));

        c_t = ggml_add(ctx0, ggml_mul(ctx0, f_t, c_t), ggml_mul(ctx0, i_t, g_t));
        h_t = ggml_mul(ctx0, o_t, ggml_tanh(ctx0, c_t));

        hs = ggml_set_1d(ctx0, hs, h_t, t*hs->nb[1]);
    }

    hs = ggml_cont(ctx0, ggml_transpose(ctx0, hs));

    return hs;
}

static struct ggml_tensor * strided_conv_transpose_1d(
                ggml_context * ctx0,
                ggml_tensor * inp,
                ggml_tensor * conv_w,
                ggml_tensor * conv_b,
                        int   stride) {
    int kernel_size   = conv_w->ne[0];
    int padding_total = kernel_size - stride;

    struct ggml_tensor * dst = ggml_conv_transpose_1d(ctx0, conv_w, inp, stride, 0, 1);

    // add bias
    dst = ggml_transpose(ctx0, dst);
    dst = ggml_add(ctx0, ggml_repeat(ctx0, conv_b, dst), dst);
    dst = ggml_cont(ctx0, ggml_transpose(ctx0, dst));

    int padding_right = ceilf(padding_total);
    int padding_left = padding_total - padding_right;

    struct ggml_tensor * unpadded = unpad_1d(ctx0, dst, padding_left, padding_right);
    unpadded = ggml_cont(ctx0, unpadded);

    return unpadded;
}

bool encodec_model_load(const std::string& fname, encodec_model& model) {
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());

    auto infile = std::ifstream(fname, std::ios::binary);
    if (!infile) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic (i.e. ggml signature in hex format)
    {
        uint32_t magic;
        read_safe(infile, magic);
        if (magic != ENCODEC_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    auto & ctx = model.ctx;
    size_t ctx_size = 0;

    // Evaluating context size
    {
        const auto & hparams = model.hparams;

        const int in_channels   = hparams.in_channels;
        const int hidden_dim    = hparams.hidden_dim;
        const int n_filters     = hparams.n_filters;
        const int kernel_size   = hparams.kernel_size;
        const int n_q           = hparams.n_q;
        const int *ratios       = hparams.ratios;

        // encoder
        {
            // initial conv1d layer
            ctx_size += in_channels*n_filters*kernel_size*ggml_type_size(GGML_TYPE_F32);  // weight
            ctx_size +=                         n_filters*ggml_type_size(GGML_TYPE_F32);  //bias

            // resnet blocks
            ctx_size +=                  3*4*16*n_filters*ggml_type_size(GGML_TYPE_F32);  // upper bound on w_g, w_v and bias

            //downsampling blocks
            ctx_size += 3*4*16*n_filters*16*n_filters*2*ratios[0]*2*ggml_type_size(GGML_TYPE_F32);  // upper bound on w_g, w_v and bias

            // lstm
            ctx_size +=             2*16*n_filters*16*n_filters*2*2*ggml_type_size(GGML_TYPE_F32); // weights
            ctx_size +=                          4*16*n_filters*2*2*ggml_type_size(GGML_TYPE_F32); // bias

            // final conv
            ctx_size +=       3*16*n_filters*hidden_dim*kernel_size*ggml_type_size(GGML_TYPE_F32); // upper bound on w_g, w_v and bias
        }

        // decoder mirrors the encoder (same number of parameter), just double context size
        ctx_size *= 2;

        // quantizer
        {
            ctx_size +=                   n_q; // inited
            ctx_size +=              n_q*1024; // cluster_size
            ctx_size += 2*n_q*hidden_dim*1024; // embed and embed_avg
        }

        ctx_size += 10ull*MB;  // object overhead
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /* .mem_size   = */   ctx_size,
            /* .mem_buffer = */   NULL,
            /* .no_alloc   = */   false,
        };

        model.ctx = ggml_init(params);
        if(!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int in_channels   = hparams.in_channels;
        const int hidden_dim    = hparams.hidden_dim;
        const int n_filters     = hparams.n_filters;
        const int kernel_size   = hparams.kernel_size;
        const int res_kernel_sz = hparams.residual_kernel_size;
        const int n_q           = hparams.n_q;
        const int *ratios       = hparams.ratios;
        const int n_bins        = hparams.n_bins;

        // encoder
        {
            model.encoder.blocks.resize(4);

            int mult = 1;  // scaling factor for hidden size

            model.encoder.init_conv_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, in_channels, mult*n_filters);
            model.encoder.init_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters);

            model.tensors["encoder.model.0.conv.conv.weight"] = model.encoder.init_conv_w;
            model.tensors["encoder.model.0.conv.conv.bias"]   = model.encoder.init_conv_b;

            for (int i = 0; i < 4; i++) {
                // conv1
                model.encoder.blocks[i].conv_1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, res_kernel_sz, mult*n_filters, mult*n_filters/2);
                model.encoder.blocks[i].conv_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/2);

                model.tensors["encoder.model." + std::to_string(3*i+1) + ".block.1.conv.conv.weight"] = model.encoder.blocks[i].conv_1_w;
                model.tensors["encoder.model." + std::to_string(3*i+1) + ".block.1.conv.conv.bias"]   = model.encoder.blocks[i].conv_1_b;

                // conv2
                model.encoder.blocks[i].conv_2_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, mult*n_filters/2, mult*n_filters);
                model.encoder.blocks[i].conv_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters);

                model.tensors["encoder.model." + std::to_string(3*i+1) + ".block.3.conv.conv.weight"] = model.encoder.blocks[i].conv_2_w;
                model.tensors["encoder.model." + std::to_string(3*i+1) + ".block.3.conv.conv.bias"]   = model.encoder.blocks[i].conv_2_b;

                // shortcut conv
                model.encoder.blocks[i].conv_sc_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, mult*n_filters, mult*n_filters);
                model.encoder.blocks[i].conv_sc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters);

                model.tensors["encoder.model." + std::to_string(3*i+1) + ".shortcut.conv.conv.weight"] = model.encoder.blocks[i].conv_sc_w;
                model.tensors["encoder.model." + std::to_string(3*i+1) + ".shortcut.conv.conv.bias"]   = model.encoder.blocks[i].conv_sc_b;

                // downsampling
                model.encoder.blocks[i].ds_conv_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2*ratios[3-i], mult*n_filters, mult*n_filters*2);
                model.encoder.blocks[i].ds_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters*2);

                model.tensors["encoder.model." + std::to_string(3*(i+1)) + ".conv.conv.weight"] = model.encoder.blocks[i].ds_conv_w;
                model.tensors["encoder.model." + std::to_string(3*(i+1)) + ".conv.conv.bias"]   = model.encoder.blocks[i].ds_conv_b;

                mult *= 2;
            }

            // LSTM
            model.encoder.lstm.l0_ih_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mult*n_filters, 4*mult*n_filters);
            model.encoder.lstm.l1_ih_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mult*n_filters, 4*mult*n_filters);

            model.tensors["encoder.model.13.lstm.weight_ih_l0"] = model.encoder.lstm.l0_ih_w;
            model.tensors["encoder.model.13.lstm.weight_ih_l1"] = model.encoder.lstm.l1_ih_w;

            model.encoder.lstm.l0_hh_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mult*n_filters, 4*mult*n_filters);
            model.encoder.lstm.l1_hh_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mult*n_filters, 4*mult*n_filters);

            model.tensors["encoder.model.13.lstm.weight_hh_l0"] = model.encoder.lstm.l0_hh_w;
            model.tensors["encoder.model.13.lstm.weight_hh_l1"] = model.encoder.lstm.l1_hh_w;

            model.encoder.lstm.l0_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);
            model.encoder.lstm.l1_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);

            model.tensors["encoder.model.13.lstm.bias_ih_l0"] = model.encoder.lstm.l0_ih_b;
            model.tensors["encoder.model.13.lstm.bias_ih_l1"] = model.encoder.lstm.l1_ih_b;

            model.encoder.lstm.l0_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);
            model.encoder.lstm.l1_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);

            model.tensors["encoder.model.13.lstm.bias_hh_l0"] = model.encoder.lstm.l0_hh_b;
            model.tensors["encoder.model.13.lstm.bias_hh_l1"] = model.encoder.lstm.l1_hh_b;

            // final conv
            model.encoder.final_conv_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, mult*n_filters, hidden_dim);
            model.encoder.final_conv_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);

            model.tensors["encoder.model.15.conv.conv.weight"] = model.encoder.final_conv_w;
            model.tensors["encoder.model.15.conv.conv.bias"]     = model.encoder.final_conv_b;
        }

        // decoder
        {
            model.decoder.blocks.resize(4);

            int mult = 16;  // 2**len(ratios)

            model.decoder.init_conv_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, hidden_dim, mult*n_filters);
            model.decoder.init_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters);

            model.tensors["decoder.model.0.conv.conv.weight"] = model.decoder.init_conv_w;
            model.tensors["decoder.model.0.conv.conv.bias"]   = model.decoder.init_conv_b;

            // LSTM
            model.decoder.lstm.l0_ih_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mult*n_filters, 4*mult*n_filters);
            model.decoder.lstm.l1_ih_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mult*n_filters, 4*mult*n_filters);

            model.tensors["decoder.model.1.lstm.weight_ih_l0"] = model.decoder.lstm.l0_ih_w;
            model.tensors["decoder.model.1.lstm.weight_ih_l1"] = model.decoder.lstm.l1_ih_w;

            model.decoder.lstm.l0_hh_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mult*n_filters, 4*mult*n_filters);
            model.decoder.lstm.l1_hh_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mult*n_filters, 4*mult*n_filters);

            model.tensors["decoder.model.1.lstm.weight_hh_l0"] = model.decoder.lstm.l0_hh_w;
            model.tensors["decoder.model.1.lstm.weight_hh_l1"] = model.decoder.lstm.l1_hh_w;

            model.decoder.lstm.l0_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);
            model.decoder.lstm.l1_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);

            model.tensors["decoder.model.1.lstm.bias_ih_l0"] = model.decoder.lstm.l0_ih_b;
            model.tensors["decoder.model.1.lstm.bias_ih_l1"] = model.decoder.lstm.l1_ih_b;

            model.decoder.lstm.l0_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);
            model.decoder.lstm.l1_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);

            model.tensors["decoder.model.1.lstm.bias_hh_l0"] = model.decoder.lstm.l0_hh_b;
            model.tensors["decoder.model.1.lstm.bias_hh_l1"] = model.decoder.lstm.l1_hh_b;

            for (int i = 0; i < 4; i++) {
                // upsampling
                model.decoder.blocks[i].us_conv_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ratios[i]*2, mult*n_filters/2, mult*n_filters);
                model.decoder.blocks[i].us_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/2);

                model.tensors["decoder.model." + std::to_string(3*(i+1)) + ".convtr.convtr.weight"] = model.decoder.blocks[i].us_conv_w;
                model.tensors["decoder.model." + std::to_string(3*(i+1)) + ".convtr.convtr.bias"]   = model.decoder.blocks[i].us_conv_b;

                // conv1
                model.decoder.blocks[i].conv_1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, res_kernel_sz, mult*n_filters/2, mult*n_filters/4);
                model.decoder.blocks[i].conv_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/4);

                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".block.1.conv.conv.weight"] = model.decoder.blocks[i].conv_1_w;
                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".block.1.conv.conv.bias"]     = model.decoder.blocks[i].conv_1_b;

                // conv2
                model.decoder.blocks[i].conv_2_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, mult*n_filters/4, mult*n_filters/2);
                model.decoder.blocks[i].conv_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/2);

                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".block.3.conv.conv.weight"] = model.decoder.blocks[i].conv_2_w;
                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".block.3.conv.conv.bias"]   = model.decoder.blocks[i].conv_2_b;

                // shortcut
                model.decoder.blocks[i].conv_sc_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, mult*n_filters/2, mult*n_filters/2);
                model.decoder.blocks[i].conv_sc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/2);

                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".shortcut.conv.conv.weight"] = model.decoder.blocks[i].conv_sc_w;
                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".shortcut.conv.conv.bias"]   = model.decoder.blocks[i].conv_sc_b;

                mult /= 2;
            }

            model.decoder.final_conv_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, n_filters, in_channels);
            model.decoder.final_conv_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

            model.tensors["decoder.model.15.conv.conv.weight"] = model.decoder.final_conv_w;
            model.tensors["decoder.model.15.conv.conv.bias"]   = model.decoder.final_conv_b;
        }

        // quantizer
        {
            model.quantizer.blocks.resize(n_q);

            for (int i = 0; i < n_q; i++) {
                model.quantizer.blocks[i].inited       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                model.quantizer.blocks[i].cluster_size = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_bins);
                model.quantizer.blocks[i].embed        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, n_bins);
                model.quantizer.blocks[i].embed_avg    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, n_bins);

                model.tensors["quantizer.vq.layers." + std::to_string(i) + "._codebook.inited"]       = model.quantizer.blocks[i].inited;
                model.tensors["quantizer.vq.layers." + std::to_string(i) + "._codebook.cluster_size"] = model.quantizer.blocks[i].cluster_size;
                model.tensors["quantizer.vq.layers." + std::to_string(i) + "._codebook.embed"]        = model.quantizer.blocks[i].embed;
                model.tensors["quantizer.vq.layers." + std::to_string(i) + "._codebook.embed_avg"]    = model.quantizer.blocks[i].embed_avg;
            }
        }

    }

    // load weights
    {
        size_t total_size = 0;
        model.n_loaded    = 0;

        while(true) {
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
            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            infile.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            printf("%48s - [%5d, %5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ne[2], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);

            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        fprintf(stderr, "%s: model size    = %7.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    infile.close();

    return true;
}

static struct ggml_cgraph * encodec_build_graph(
                     encodec_context & ectx, 
            const std::vector<float> & inp_audio) {
    const int32_t audio_length = inp_audio.size();
    
    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ ectx.buf_compute.size(),
        /*.mem_buffer =*/ ectx.buf_compute.data(),
        /*.no_alloc   =*/ true, // skip allocating as we use ggml_alloc to allocate exact memory requirements
    };

    struct ggml_context * ctx0 = ggml_init(ggml_params);
    struct ggml_cgraph  * gf   = ggml_new_graph(ctx0);

    struct ggml_tensor * inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, audio_length);
    ggml_allocr_alloc(ectx.allocr, inp);
    if (!ggml_allocr_measure(ectx.allocr)) {
        memcpy(inp->data, inp_audio.data(), audio_length*ggml_element_size(inp));
    }

    // encoder
    struct ggml_tensor * encoded_inp;
    {
        const auto & hparams = model.hparams;

        const int * ratios      = hparams.ratios;
        const int kernel_size   = hparams.kernel_size;
        const int res_kernel_sz = hparams.residual_kernel_size;
        const int stride        = hparams.stride;

        struct ggml_tensor * inpL = strided_conv_1d(
            ctx0, inp, model.encoder.init_conv_w, model.encoder.init_conv_b, stride);

        for (int layer_ix = 0; layer_ix < 4; layer_ix++) {
            encodec_encoder_block block = model.encoder.blocks[layer_ix];

            struct ggml_tensor * current = inpL;

            // shortcut
            struct ggml_tensor * shortcut = strided_conv_1d(
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
                ctx0, inpL, block.ds_conv_w, block.ds_conv_b, ratios[3-layer_ix]);
        }

        // lstm
        {
            struct ggml_tensor * cur = inpL;

            const encodec_lstm lstm = model.encoder.lstm;

            // first lstm layer
            struct ggml_tensor * hs1 = forward_pass_lstm_unilayer(
                ctx0, cur, lstm.l0_ih_w, lstm.l0_hh_w, lstm.l0_ih_b, lstm.l0_hh_b);

            // second lstm layer
            struct ggml_tensor * out = forward_pass_lstm_unilayer(
                ctx0, hs1, lstm.l1_ih_w, lstm.l1_hh_w, lstm.l1_ih_b, lstm.l1_hh_b);

            inpL = ggml_add(ctx0, inpL, out);
        }

        // final conv
        {
            inpL = ggml_elu(ctx0, inpL);

            encoded_inp = strided_conv_1d(
                ctx0, inpL, model.encoder.final_conv_w, model.encoder.final_conv_b, stride);
        }
    }

    // quantizer (encode)
    struct ggml_tensor * codes;
    {
        const auto & hparams = model.hparams;
        // originally, n_q = n_q or len(self.layers)
        // for this model, n_q is at most 32, but the implementation we are comparing
        // our model against has only 16, hence we hardcode 16 as n_q for now.
        // const int n_q = hparams.n_q;
        const int n_q = 16;

        const int seq_length = encoded_inp->ne[0];
        codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, seq_length, n_q);

        struct ggml_tensor * inpL = ggml_cont(ctx0, ggml_transpose(ctx0, encoded_inp));
        struct ggml_tensor * residual = inpL;
        struct ggml_tensor * indices;

        for (int i = 0; i < n_q; i++) {
            encodec_quant_block block = model.quantizer.blocks[i];

            // compute distance
            // [seq_length, n_bins]
            struct ggml_tensor * dp = ggml_scale(
                    ctx0, ggml_mul_mat(ctx0, block.embed, residual), ggml_new_f32(ctx0, -2.0f));

            // [n_bins]
            struct ggml_tensor * sqr_embed     = ggml_sqr(ctx0, block.embed);
            struct ggml_tensor * sqr_embed_nrm = ggml_sum_rows(ctx0, sqr_embed);

            // [seq_length]
            struct ggml_tensor * sqr_inp     = ggml_sqr(ctx0, residual);
            struct ggml_tensor * sqr_inp_nrm = ggml_sum_rows(ctx0, sqr_inp);

            // [seq_length, n_bins]
            struct ggml_tensor * dist = ggml_add(ctx0, ggml_repeat(ctx0, sqr_inp_nrm, dp), dp);
            dist = ggml_add(ctx0, ggml_repeat(ctx0, ggml_transpose(ctx0, sqr_embed_nrm), dist), dist);
            dist = ggml_scale(ctx0, dist, ggml_new_f32(ctx0, -1.0f));

            // take the argmax over the column dimension
            // [seq_length]
            indices = ggml_argmax(ctx0, dist);

            // look up in embedding table
            struct ggml_tensor * quantized = ggml_get_rows(ctx0, block.embed, indices);

            residual = ggml_sub(ctx0, residual, quantized);

            codes = ggml_set_1d(ctx0, codes, indices, i*codes->nb[1]);
        }

    }

    // quantizer (decode)
    struct ggml_tensor * quantized_out;
    {
        const auto & hparams = model.hparams;
        const int hidden_dim = hparams.hidden_dim;

        const int seq_length = codes->ne[0];
        const int n_q        = codes->ne[1];

        quantized_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, seq_length);
        quantized_out = ggml_set_zero(quantized_out);

        for (int i = 0; i < n_q; i++) {
            encodec_quant_block block = model.quantizer.blocks[i];

            struct ggml_tensor * indices   = ggml_view_1d(ctx0, codes, seq_length, i*codes->nb[1]);
            struct ggml_tensor * quantized = ggml_get_rows(ctx0, block.embed, indices);

            quantized_out = ggml_add(ctx0, quantized_out, quantized);
        }

        quantized_out = ggml_cont(ctx0, ggml_transpose(ctx0, quantized_out));
    }

    // decoder
    struct ggml_tensor * decoded_inp;
    struct ggml_tensor * out;
    {
        const auto & hparams = model.hparams;

        const int * ratios      = hparams.ratios;
        const int kernel_size   = hparams.kernel_size;
        const int res_kernel_sz = hparams.residual_kernel_size;
        const int stride        = hparams.stride;

        struct ggml_tensor * inpL = strided_conv_1d(
            ctx0, quantized_out, model.decoder.init_conv_w, model.decoder.init_conv_b, stride);

        // lstm
        {
            struct ggml_tensor * cur = inpL;

            const encodec_lstm lstm = model.decoder.lstm;

            // first lstm layer
            struct ggml_tensor * hs1 = forward_pass_lstm_unilayer(
                ctx0, cur, lstm.l0_ih_w, lstm.l0_hh_w, lstm.l0_ih_b, lstm.l0_hh_b);

            // second lstm layer
            struct ggml_tensor * out = forward_pass_lstm_unilayer(
                ctx0, hs1, lstm.l1_ih_w, lstm.l1_hh_w, lstm.l1_ih_b, lstm.l1_hh_b);

            inpL = ggml_add(ctx0, inpL, out);
        }

        for (int layer_ix = 0; layer_ix < 4; layer_ix++) {
            encodec_decoder_block block = model.decoder.blocks[layer_ix];

            // upsampling layers
            inpL = ggml_elu(ctx0, inpL);

            inpL = strided_conv_transpose_1d(
                ctx0, inpL, block.us_conv_w, block.us_conv_b, ratios[layer_ix]);

            struct ggml_tensor * current = inpL;
            
            // shortcut
            struct ggml_tensor * shortcut = strided_conv_1d(
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
        {
            inpL = ggml_elu(ctx0, inpL);

            decoded_inp = strided_conv_1d(
                ctx0, inpL, model.decoder.final_conv_w, model.decoder.final_conv_b, stride);
        }

        out = decoded_inp;
    }

    out = ggml_cpy(ectx.ctx_audio, out, ectx.reconstructed_audio);

    ggml_build_forward_expand(gf, out);
    ggml_disconnect_node_from_graph(ectx.reconstructed_audio);

    ggml_free(ctx0);

    return gf;
}

static bool encodec_model_eval(
                std::vector<float> & raw_audio, 
                   encodec_context & ectx, 
                               int   n_threads) {
    if (!ectx.model) {
        return false;
    }

    const int64_t t_start_ms = ggml_time_ms();

    fprintf(stderr, "%s: raw audio (t=%zu)\n", __func__, raw_audio.size());

    static const size_t buf_size = 256u*1024*1024;

    if (ectx.ctx_audio) {
        ggml_free(ectx.ctx_audio);
        ectx.ctx_audio = {};
    }

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    ectx.ctx_audio = ggml_init(ggml_params);

    ectx.reconstructed_audio = ggml_new_tensor_1d(ectx.ctx_audio, GGML_TYPE_F32, raw_audio.size());

    // reconstruct the audio
    ectx.buf_compute.resize(ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead());
    ectx.allocr = ggml_allocr_new_measure(TENSOR_ALIGNMENT);
    struct ggml_cgraph * gf_measure = encodec_build_graph(ectx, raw_audio);
    if (!gf_measure) {
        fprintf(stderr, "%s: failed to build graph\n", __func__);
        return false;
    }

    size_t alloc_size = ggml_allocr_alloc_graph(ectx.allocr, gf_measure) + TENSOR_ALIGNMENT;
    ggml_allocr_free(ectx.allocr);

    // recreate allocator with exact memory requirements
    ectx.buf_alloc.resize(alloc_size);
    ectx.allocr = ggml_allocr_new(ectx.buf_alloc.data(), ectx.buf_alloc.size(), TENSOR_ALIGNMENT);

    // compute the graph with the measured exact memory requirements from above
    ggml_allocr_reset(ectx.allocr);

    struct ggml_cgraph * gf = encodec_build_graph(ectx, raw_audio);
    if (!gf) {
        fprintf(stderr, "%s: failed to build graph\n", __func__);
        return false;
    }

    ggml_allocr_alloc_graph(ectx.allocr, gf);

    ggml_graph_compute_helper(ectx.work_buffer, gf, n_threads);

    ggml_allocr_free(ectx.allocr);
    ectx.allocr = NULL;
    ectx.work_buffer.clear();

    ectx.t_compute_ms = ggml_time_ms() - t_start_ms;

    return true;
}
