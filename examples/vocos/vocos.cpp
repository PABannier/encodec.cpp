/* This demonstrates how to use the Vocos encoder with Encodec code features
to reconstruct an audio.

Author: Pierre-Antoine Bannier
*/
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <thread>

#include "encodec.h"
#include "common.h"

#define VOCOS_FILE_MAGIC 'ggml'

static const size_t MB = 1024 * 1024;

static void print_tensor(struct ggml_tensor *t) {
    printf("tensor %s: %lld %lld %lld\n", t->name, t->ne[0], t->ne[1], t->ne[2]);
}

struct vocos_params {
    // Number of threads used for inference
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    // Target bandwidth
    int32_t bandwidth_id = 2;

    // Input location
    std::string input_path = "input.wav";

    // Vocos weights location
    std::string vocos_model_path = "./vocos/ggml-model.bin";

    // Encodec weights location
    std::string encodec_model_path = "./encodec/ggml-model.bin";

    // Output location
    std::string output_path = "output.wav";
};

struct vocos_hparams {
    // Number of input channels in backbone
    int32_t input_channels;
    // Inner dimension in backbone
    int32_t dim;
    // Intermediate dimension in backbone
    int32_t dim_intermediate;
    // Number of layers in backbone
    int32_t n_layers;
    // Number of codes
    int32_t adanorm_num_embeddings;
    // Dimension in head
    int32_t head_dim;
    // Number of FFT bins
    int32_t n_fft;
    // Hop length
    int32_t hop_length;

    // Bandwidth identifier
    int32_t bandwidth_id;

    // File type of model weights
    int32_t ftype;
};

struct vocos_backbone_layer {
    struct ggml_tensor * dwconv_w;
    struct ggml_tensor * dwconv_b;

    struct ggml_tensor * gamma;

    struct ggml_tensor * norm_scale;
    struct ggml_tensor * norm_shift;

    struct ggml_tensor * pwconv1_w;
    struct ggml_tensor * pwconv1_b;

    struct ggml_tensor * pwconv2_w;
    struct ggml_tensor * pwconv2_b;
};

struct vocos_backbone {
    struct ggml_tensor * embed_w;
    struct ggml_tensor * embed_b;

    struct ggml_tensor * norm_scale;
    struct ggml_tensor * norm_shift;

    struct ggml_tensor * final_ln_w;
    struct ggml_tensor * final_ln_b;

    std::vector<struct vocos_backbone_layer> layers;
};

struct vocos_feature_extractor {
    struct ggml_tensor *codebook_weights;
};

struct vocos_head {
    struct ggml_tensor *istft_window;
    struct ggml_tensor *proj_out_w;
    struct ggml_tensor *proj_out_b;
};

struct vocos_model {
    struct vocos_hparams hparams;

    struct vocos_backbone backbone;
    struct vocos_feature_extractor feature_extractor;
    struct vocos_head head;

    // context
    struct ggml_context * ctx;
    int n_loaded;

    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer_w;

    std::map<std::string, struct ggml_tensor *> tensors;
};

struct vocos_statistics {
    // The time taken to load the model.
    int64_t t_load_us;
    // The time taken to compute the model.
    int64_t t_compute_us;
};

struct vocos_context {
    struct vocos_model model;

    // buffer for model evaluation
    ggml_backend_buffer_t buf_compute;

    // custom allocator
    struct ggml_allocr * allocr = NULL;

    // intermediate steps
    struct ggml_tensor * features_t  = NULL;
    struct ggml_tensor * out_audio_t = NULL;

    std::vector<float> features ;
    std::vector<float> out_audio;

    // statistics
    struct vocos_statistics stats;

    // parameters
    int32_t n_threads;
    std::string encodec_path;
};

typedef enum {
    // Run the end-to-end encoder-decoder pipeline
    full = 0,
    // Encode an audio
    encode = 1,
    // Decode an audio from codes
    decode = 2,
} vocos_run_mode;

template<typename T>
static void read_safe(std::ifstream &fin, T &dest) {
    fin.read((char *)&dest, sizeof(T));
}

const struct vocos_statistics* vocos_get_statistics(struct vocos_context *vctx) {
    if (!vctx) {
        fprintf(stderr, "%s: null context\n", __func__);
        return nullptr;
    }
    return &vctx->stats;
}

bool vocos_load_model_weights(std::ifstream &fin, struct vocos_model &model) {
    // verify magic
    {
        uint32_t magic;
        read_safe(fin, magic);
        if (magic != VOCOS_FILE_MAGIC) {
            std::cerr << "Invalid file magic" << std::endl;
            return false;
        }
    }

    // load hparams
    {
        auto &hparams = model.hparams;

        read_safe(fin, hparams.input_channels);
        read_safe(fin, hparams.dim);
        read_safe(fin, hparams.dim_intermediate);
        read_safe(fin, hparams.n_layers);
        read_safe(fin, hparams.adanorm_num_embeddings);
        read_safe(fin, hparams.head_dim);
        read_safe(fin, hparams.n_fft);
        read_safe(fin, hparams.hop_length);
        read_safe(fin, hparams.ftype);

        printf("%s: input_channels = %d\n",         __func__, hparams.input_channels);
        printf("%s: dim = %d\n",                    __func__, hparams.dim);
        printf("%s: dim_intermediate = %d\n",       __func__, hparams.dim_intermediate);
        printf("%s: n_layers = %d\n",               __func__, hparams.n_layers);
        printf("%s: adanorm_num_embeddings = %d\n", __func__, hparams.adanorm_num_embeddings);
        printf("%s: head_dim = %d\n",               __func__, hparams.head_dim);
        printf("%s: n_fft = %d\n",                  __func__, hparams.n_fft);
        printf("%s: hop_length = %d\n",             __func__, hparams.hop_length);
        printf("%s: ftype = %d\n",                  __func__, hparams.ftype);
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        std::cerr << "Invalid model file (bad ftype value " << model.hparams.ftype << ")" << std::endl;
        return false;
    }

    auto &ctx = model.ctx;

    size_t buffer_size = 0;
    size_t n_tensors = 0;

    // Evaluating context size
    {
        const auto &hparams = model.hparams;

        const int input_channels         = hparams.input_channels;
        const int dim                    = hparams.dim;
        const int dim_intermediate       = hparams.dim_intermediate;
        const int n_layers               = hparams.n_layers;
        const int adanorm_num_embeddings = hparams.adanorm_num_embeddings;
        const int head_dim               = hparams.head_dim;
        const int n_fft                  = hparams.n_fft;
        const int hop_length             = hparams.hop_length;

        // backbone
        buffer_size += input_channels * dim * 7 * ggml_type_size(wtype);                     // embed_w
        buffer_size += dim * ggml_type_size(GGML_TYPE_F32);                                  // embed_b

        buffer_size += 2 * dim * ggml_type_size(GGML_TYPE_F32);                              // final_layer_norm
        buffer_size += 2 * dim * adanorm_num_embeddings * ggml_type_size(GGML_TYPE_F32);     // norm_scale and norm_shift

        buffer_size += n_layers * dim * dim * 7 * ggml_type_size(wtype);                     // dwconv_w
        buffer_size += n_layers * dim           * ggml_type_size(GGML_TYPE_F32);             // dwconv_b
        buffer_size += n_layers * 2 * dim       * ggml_type_size(GGML_TYPE_F32);             // gamma
        buffer_size += n_layers * dim * adanorm_num_embeddings * ggml_type_size(wtype);      // norm_scale and norm_shift
        buffer_size += n_layers * 2 * dim * dim_intermediate * ggml_type_size(wtype);        // pwconv1_w and pwconv2_w
        buffer_size += n_layers * dim * ggml_type_size(GGML_TYPE_F32);                       // pwconv1_b
        buffer_size += n_layers * dim_intermediate * ggml_type_size(GGML_TYPE_F32);          // pwconv2_b

        n_tensors += 6 + n_layers * 9;

        // feature extactor
        buffer_size += 16384 * input_channels * ggml_type_size(GGML_TYPE_F32); // TODO(PAB): hardcoded value!
        n_tensors++;

        // head
        buffer_size += dim * (n_fft + 2) * ggml_type_size(wtype);   // proj_out_w
        buffer_size += (n_fft + 2) * ggml_type_size(GGML_TYPE_F32); // proj_out_b
        buffer_size += n_fft * ggml_type_size(GGML_TYPE_F32);   // istft_window

        n_tensors += 3;

        buffer_size += 10ull * MB; // object overhead

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
            std::cerr << __func__ << ": ggml_init() failed" << std::endl;
            return false;
        }
    }

    if (!model.backend) {
        // fallback to CPU backend
        std::cerr << __func__ << ": using CPU backend" << std::endl;
        model.backend = ggml_backend_cpu_init();
    }

    if (!model.backend) {
        std::cerr << __func__ << ": ggml_backend_cpu_init() failed" << std::endl;
        return false;
    }

    // allocate weights buffer
    model.buffer_w = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // prepare memory for the weights
    {
        const auto &hparams = model.hparams;

        const int input_channels         = hparams.input_channels;
        const int dim                    = hparams.dim;
        const int dim_intermediate       = hparams.dim_intermediate;
        const int n_layers               = hparams.n_layers;
        const int adanorm_num_embeddings = hparams.adanorm_num_embeddings;
        const int head_dim               = hparams.head_dim;
        const int n_fft                  = hparams.n_fft;
        const int hop_length             = hparams.hop_length;

        // backbone
        {
            model.backbone.layers.resize(n_layers);

            model.backbone.embed_w = ggml_new_tensor_3d(ctx, wtype, 7, input_channels, dim);
            model.backbone.embed_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);

            model.tensors["backbone/embed/w"] = model.backbone.embed_w;
            model.tensors["backbone/embed/b"] = model.backbone.embed_b;

            model.backbone.norm_scale = ggml_new_tensor_2d(ctx, wtype, dim, adanorm_num_embeddings);
            model.backbone.norm_shift = ggml_new_tensor_2d(ctx, wtype, dim, adanorm_num_embeddings);

            model.tensors["backbone/norm/scale/w"] = model.backbone.norm_scale;
            model.tensors["backbone/norm/shift/w"] = model.backbone.norm_shift;

            model.backbone.final_ln_w = ggml_new_tensor_1d(ctx, wtype, dim);
            model.backbone.final_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);

            model.tensors["backbone/final_layer_norm/w"] = model.backbone.final_ln_w;
            model.tensors["backbone/final_layer_norm/b"] = model.backbone.final_ln_b;

            for (int i = 0; i < n_layers; i++) {
                auto &layer = model.backbone.layers[i];

                layer.dwconv_w = ggml_new_tensor_3d(ctx, wtype, 7, 1, 384);
                layer.dwconv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);

                model.tensors["backbone/convnext/" + std::to_string(i) + "/dwconv/w"]   = layer.dwconv_w;
                model.tensors["backbone/convnext/" + std::to_string(i) + "/dwconv/b"]   = layer.dwconv_b;

                layer.gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);

                model.tensors["backbone/convnext/" + std::to_string(i) + "/gamma"]      = layer.gamma;

                layer.norm_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, adanorm_num_embeddings);
                layer.norm_shift = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, adanorm_num_embeddings);

                model.tensors["backbone/convnext/" + std::to_string(i) + "/norm/scale"] = layer.norm_scale;
                model.tensors["backbone/convnext/" + std::to_string(i) + "/norm/shift"] = layer.norm_shift;

                layer.pwconv1_w = ggml_new_tensor_2d(ctx, wtype, dim, dim_intermediate);
                layer.pwconv1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_intermediate);

                model.tensors["backbone/convnext/" + std::to_string(i) + "/pwconv/1/w"] = layer.pwconv1_w;
                model.tensors["backbone/convnext/" + std::to_string(i) + "/pwconv/1/b"] = layer.pwconv1_b;

                layer.pwconv2_w = ggml_new_tensor_2d(ctx, wtype, dim_intermediate, dim);
                layer.pwconv2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);

                model.tensors["backbone/convnext/" + std::to_string(i) + "/pwconv/2/w"] = layer.pwconv2_w;
                model.tensors["backbone/convnext/" + std::to_string(i) + "/pwconv/2/b"] = layer.pwconv2_b;
            }
        }

        // feature extractor
        {
            // TODO (PAB): careful with hardcoded
            model.feature_extractor.codebook_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_channels, 16384);
            model.tensors["feature_extractor/codebook_weights"] = model.feature_extractor.codebook_weights;
        }

        // head
        {
            model.head.istft_window = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_fft);
            model.tensors["head/istft/window"] = model.head.istft_window;

            model.head.proj_out_w   = ggml_new_tensor_2d(ctx, wtype, dim, n_fft + 2);
            model.head.proj_out_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_fft + 2);

            model.tensors["head/out/w"] = model.head.proj_out_w;
            model.tensors["head/out/b"] = model.head.proj_out_b;
        }
    }

    // load weights
    {
        ggml_allocr *alloc = ggml_allocr_new_from_buffer(model.buffer_w);

        size_t total_size = 0;
        model.n_loaded = 0;

        std::vector<char> read_buf;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            read_safe(fin, n_dims);
            read_safe(fin, length);
            read_safe(fin, ftype);

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[3] = { 1, 1, 1 };
            for (int i = 0; i < n_dims; i++) {
                read_safe(fin, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> buf(length);
            fin.read(&buf[0], buf.size());
            name.assign(&buf[0], buf.size());

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                std::cerr << "Unknown tensor name: " << name << std::endl;
                return false;
            }

            auto tensor = model.tensors[name.data()];
            ggml_set_name(tensor, name.c_str());
            if (ggml_nelements(tensor) != nelements) {
                std::cerr << "Invalid number of elements for tensor " << name << " (" << ggml_nelements(tensor) << " != " << nelements << ")" << std::endl;
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

            ggml_allocr_alloc(alloc, tensor);

            if (ggml_backend_is_cpu(model.backend)) {
                // for the CPU and Metal backends, we can read directly into the device memory
                fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(ggml_nbytes(tensor));
                fin.read(read_buf.data(), ggml_nbytes(tensor));
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        ggml_allocr_free(alloc);
        printf("%s: model size = %8.2f MB\n", __func__, total_size / 1024.0 / 1024.0);
    }

    fin.close();

    return true;
}

struct vocos_context *vocos_load_model(const std::string model_path) {
    int64_t t_start_load_us = ggml_time_us();

    auto fin = std::ifstream(model_path, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open model file" << std::endl;
        return nullptr;
    }

    struct vocos_context *vctx = new vocos_context();

    vctx->model = vocos_model();
    if (!vocos_load_model_weights(fin, vctx->model)) {
        std::cerr << "Failed to load model weights" << std::endl;
        delete vctx;
        return nullptr;
    }

    vctx->stats.t_load_us = ggml_time_us() - t_start_load_us;

    return vctx;
}

struct ggml_tensor *vocos_ada_layer_norm(
    struct ggml_context *ctx0,
    struct ggml_tensor *inp,
    struct ggml_tensor *scale_w,
    struct ggml_tensor *shift_w,
    struct ggml_tensor *cond_embedding_id) {

    struct ggml_tensor * scale = ggml_get_rows(ctx0, scale_w, cond_embedding_id);
    struct ggml_tensor * shift = ggml_get_rows(ctx0, shift_w, cond_embedding_id);

    struct ggml_tensor * norm = ggml_norm(ctx0, inp, 1e-5 /* eps */);
    struct ggml_tensor * out = ggml_add(ctx0, ggml_mul(ctx0, norm, scale), shift);

    return out;
}

struct ggml_tensor *vocos_forward_encoder(
    struct vocos_context * vctx,
    struct ggml_context  * ctx0,
    struct ggml_tensor   * codes) {

    if (!codes) {
        fprintf(stderr, "%s: invalid codes tensor\n", __func__);
        return nullptr;
    }

    const auto & model  = vctx->model.feature_extractor;
    const auto & allocr = vctx->allocr;

    const int seq_length = codes->ne[0];
    const int        n_q = codes->ne[1];
    const int        dim = model.codebook_weights->ne[0];

    // codes: [seq_length, n_q] -> [n_q, seq_length]
    codes = ggml_transpose(ctx0, codes);

    struct ggml_tensor *features = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, dim, n_q, seq_length);
    ggml_allocr_alloc(allocr, features);

    for (int t = 0; t < seq_length; t++) {
        // [n_q]
        size_t offset = t * codes->nb[1];
        struct ggml_tensor *idxs = ggml_view_1d(ctx0, codes, n_q, offset);

        // [dim, n_q]
        struct ggml_tensor *f_t = ggml_get_rows(ctx0, model.codebook_weights, idxs);

        features = ggml_set_2d(ctx0, features, f_t, features->nb[1], t*features->nb[2]);
    }

    // [dim, n_q, seq_length] -> [n_q, dim, seq_length]
    features = ggml_cont(ctx0, ggml_permute(ctx0, features, 1, 0, 2, 3));

    // [1, dim, seq_length]
    features = ggml_sum_rows(ctx0, features);

    return features;
}

struct ggml_tensor *vocos_forward_decoder(
    struct vocos_context *vctx,
    struct ggml_context *ctx0,
    struct ggml_tensor *encoded,
    struct ggml_tensor *bandwidth_id) {
    if (!encoded) {
        fprintf(stderr, "%s: invalid input tensor\n", __func__);
        return nullptr;
    }

    const auto &model = vctx->model;
    const auto &backbone = model.backbone;
    const auto &head = model.head;

    const auto &hparams = model.hparams;
    const int n_layers = hparams.n_layers;

    // backbone

    // [dim, seq_length]
    struct ggml_tensor *emb = ggml_conv_1d(
        ctx0, backbone.embed_w, encoded, 1 /* s0 */, 3 /* p0 */, 1 /* d0 */);
    print_tensor(emb);
    print_tensor(backbone.embed_b);
    emb = ggml_add(ctx0, emb, backbone.embed_b);

    // [dim, seq_length]
    emb = vocos_ada_layer_norm(ctx0, emb, backbone.norm_scale, backbone.norm_shift, bandwidth_id);

    struct ggml_tensor *res = emb;

    for (int i = 0; i < n_layers; i++) {
        auto &layer = backbone.layers[i];

        // [dim, seq_length]
        // TODO (PAB): depth wise (groups=dim)
        struct ggml_tensor *dwconv = ggml_conv_1d(
            ctx0, layer.dwconv_w, res, 1 /* s0 */, 3 /* p0 */, 1 /* d0 */);
        dwconv = ggml_add(ctx0, dwconv, layer.dwconv_b);

        dwconv = vocos_ada_layer_norm(ctx0, dwconv, layer.norm_scale, layer.norm_shift, bandwidth_id);

        // [intermediate_dim, seq_length]
        struct ggml_tensor * pwconv1 = ggml_mul_mat(ctx0, layer.pwconv1_w, dwconv);
        pwconv1 = ggml_add(ctx0, pwconv1, layer.pwconv1_b);

        pwconv1 = ggml_gelu(ctx0, pwconv1);

        // [dim, seq_length]
        struct ggml_tensor *pwconv2 = ggml_mul_mat(ctx0, layer.pwconv2_w, pwconv1);
        pwconv2 = ggml_add(ctx0, pwconv2, layer.pwconv2_b);

        // [dim, seq_length]
        pwconv2 = ggml_mul(ctx0, pwconv2, layer.gamma);

        // [dim, seq_length], residual connection
        res = ggml_add(ctx0, res, pwconv2);
    }

    struct ggml_tensor * out = ggml_norm(ctx0, res, 1e-5 /* eps */);
    out = ggml_mul(ctx0, out, backbone.final_ln_w);
    out = ggml_add(ctx0, out, backbone.final_ln_b);

    // head
    // out = istft_head_forward(ctx0, out);

    return out;
}

struct ggml_cgraph *vocos_build_graph(
    struct vocos_context *vctx,
    const std::vector<int32_t> codes,
    const vocos_run_mode mode) {
    assert(mode == vocos_run_mode::full || mode == vocos_run_mode::encode);

    const auto & model   = vctx->model;
    const auto & hparams = model.hparams;
    const auto & allocr  = vctx->allocr;

    const int    n_q     = 8;                       // TODO (PAB): hardcoded
    const int n_bins     = 1024;                    // TODO (PAB): hardcoded
    const int seq_length = codes.size() / n_q;

    // since we are using ggml-alloc, this buffer only needs enough space to hold the
    // ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead() * GGML_MAX_NODES + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context *ctx0 = ggml_init(ggml_params);

    struct ggml_cgraph *gf = ggml_new_graph(ctx0);

    struct ggml_tensor *inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, seq_length, n_q);
    ggml_allocr_alloc(allocr, inp);

    struct ggml_tensor *bandwidth_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_allocr_alloc(allocr, bandwidth_id);

    if (!ggml_allocr_is_measure(allocr)) {
        ggml_backend_tensor_set(inp, codes.data(), 0, codes.size() * sizeof(int32_t));

        // add offsets of shape [n_q] broadcasted to inp
        // inp + offsets
        // TODO: can we ensure i / seq_length is floored?
        for (int i = 0; i < seq_length*n_q; i++) {
            int32_t v = (i / seq_length) * n_bins;
            ggml_backend_tensor_set(inp, &v, i * sizeof(int32_t), sizeof(int32_t));
        }

        ggml_backend_tensor_set(bandwidth_id, &hparams.bandwidth_id, 0, sizeof(int32_t));
    }

    struct ggml_tensor * encoded  = vocos_forward_encoder(vctx, ctx0, inp);
    struct ggml_tensor * decoded  = vocos_forward_decoder(vctx, ctx0, encoded, bandwidth_id);

    switch (mode) {
        case vocos_run_mode::full: {
            ggml_build_forward_expand(gf, decoded);
        } break;
        case vocos_run_mode::encode: {
            ggml_build_forward_expand(gf, encoded);
        } break;
        case vocos_run_mode::decode: {
            ggml_build_forward_expand(gf, decoded);
        } break;
        default: {
            fprintf(stderr, "%s: unknown run mode\n", __func__);
            return NULL;
        } break;
    }

    ggml_free(ctx0);

    vctx->features_t  = encoded;
    vctx->out_audio_t = decoded;

    return gf;
}

bool vocos_eval_internal(
    struct vocos_context *vctx,
    const std::vector<int32_t> codes,
    const int n_threads,
    const vocos_run_mode mode) {
    auto &model = vctx->model;
    auto &allocr = vctx->allocr;

    // reset the allocator to free all the memory allocated during the previous inference
    ggml_allocr_reset(allocr);

    struct ggml_cgraph *gf = vocos_build_graph(vctx, codes, mode);

    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);

    // run the computation
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf);

    return true;
}

std::vector<int32_t> get_encodec_codes(struct vocos_context *vctx, const std::vector<float> raw_audio) {
    struct encodec_context * ectx = encodec_load_model(vctx->encodec_path.c_str(), 0, 0);
    if (!ectx) {
        printf("%s: failed to load encodec model\n", __func__);
        return std::vector<int32_t>();
    }

    const auto & hparams = vctx->model.hparams;
    if (hparams.bandwidth_id < 0 || hparams.bandwidth_id > 4) {
        printf("%s: invalid bandwidth id\n", __func__);
        return std::vector<int32_t>();
    }

    encodec_set_target_bandwidth(ectx, 6);

    if (!encodec_compress_audio(ectx, raw_audio.data(), raw_audio.size(), vctx->n_threads)) {
        printf("%s: failed to compress audio\n", __func__);
        return std::vector<int32_t>();
    }

    int32_t * codes_data = encodec_get_codes(ectx);
    int n_codes = encodec_get_codes_size(ectx);
    std::vector<int32_t> codes_arr(codes_data, codes_data + n_codes);

    return codes_arr;
}

bool vocos_eval(
    struct vocos_context *vctx,
    const std::vector<float> raw_audio,
    const int n_threads,
    const vocos_run_mode mode) {
    const int64_t t_start_us = ggml_time_us();

    // Encodec forward pass, shape [n_q, T]
    // n_q depends on the bandwidth and the sample rate
    std::vector<int32_t> codes = get_encodec_codes(vctx, raw_audio);

    // allocate the compute buffer
    {
        // alignment required by the backend
        size_t align = ggml_backend_get_alignment(vctx->model.backend);
        vctx->allocr = ggml_allocr_new_measure(align);

        // create the graph for memory usage estimation
        struct ggml_cgraph *gf = vocos_build_graph(vctx, codes, mode);

        // compute the required memory
        size_t mem_size = ggml_allocr_alloc_graph(vctx->allocr, gf);

        // recreate the allocator with the required memory
        ggml_allocr_free(vctx->allocr);
        vctx->buf_compute = ggml_backend_alloc_buffer(vctx->model.backend, mem_size);
        vctx->allocr = ggml_allocr_new_from_buffer(vctx->buf_compute);

        fprintf(stderr, "%s: compute buffer size: %.2f MB\n\n", __func__, mem_size / 1024.0 / 1024.0);
    }

    if (!vocos_eval_internal(vctx, codes, n_threads, mode)) {
        fprintf(stderr, "%s: failed to run encodec eval\n", __func__);
        return false;
    }

    int32_t n_features = ggml_nelements(vctx->features_t);

    vctx->features.resize(n_features);
    ggml_backend_tensor_get(vctx->features_t, vctx->features.data(), 0, n_features * sizeof(float));

    vctx->stats.t_compute_us = ggml_time_us() - t_start_us;

    return true;
}

bool vocos_reconstruct_audio(
            struct vocos_context *vctx,
            const std::vector<float> raw_audio,
            int n_threads) {
    if (!vocos_eval(vctx, raw_audio, n_threads, vocos_run_mode::full)) {
        std::cerr << "Failed to evaluate model" << std::endl;
        return false;
    }

    return true;
}

void vocos_free(struct vocos_context *vctx) {
    if (!vctx) {
        return;
    }

    if (vctx->model.ctx) {
        ggml_free(vctx->model.ctx);
    }

    if (vctx->buf_compute) {
        ggml_backend_buffer_free(vctx->buf_compute);
    }

    ggml_backend_buffer_free(vctx->model.buffer_w);
    ggml_backend_free(vctx->model.backend);

    delete vctx;
}

void vocos_print_usage(char ** argv, const vocos_params &params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help             show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N      number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -b N, --bandwidth_id N Target bandwidth identifier (default: %d)\n", params.bandwidth_id);
    fprintf(stderr, "  -vm FNAME, --vocos_model FNAME\n");
    fprintf(stderr, "                         Vocos model path (default: %s)\n", params.vocos_model_path.c_str());
    fprintf(stderr, "  -em FNAME, --encodec_model FNAME\n");
    fprintf(stderr, "                         Encodec model path (default: %s)\n", params.encodec_model_path.c_str());
    fprintf(stderr, "  -i FNAME, --input FNAME\n");
    fprintf(stderr, "                         original audio wav (default: %s)\n", params.input_path.c_str());
    fprintf(stderr, "  -o FNAME, --outwav FNAME\n");
    fprintf(stderr, "                         output generated wav (default: %s)\n", params.output_path.c_str());
    fprintf(stderr, "\n");
}

int vocos_params_parse(int argc, char ** argv, vocos_params &params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-b" || arg == "--bandwidth_id") {
            params.bandwidth_id = std::stoi(argv[++i]);
        } else if (arg == "-vm" || arg == "--vocos_model") {
            params.vocos_model_path = argv[++i];
        } else if (arg == "-em" || arg == "--encodec_model") {
            params.encodec_model_path = argv[++i];
        } else if (arg == "-o" || arg == "--outwav") {
            params.output_path = argv[++i];
        } else if (arg == "-i" || arg == "--input") {
            params.input_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            vocos_print_usage(argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            vocos_print_usage(argv, params);
            exit(0);
        }
    }

    return 0;
}

int main(int argc, char **argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    vocos_params params;

    if (vocos_params_parse(argc, argv, params) > 0) {
        fprintf(stderr, "%s: Could not parse arguments\n", __func__);
        return 1;
    }

    struct vocos_context *vctx = vocos_load_model(params.vocos_model_path);
    if (!vctx) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    vctx->encodec_path = params.encodec_model_path;
    vctx->model.hparams.bandwidth_id = params.bandwidth_id;

    // read audio from disk
    std::vector<float> original_audio_arr;
    if (!read_wav_from_disk(params.input_path, original_audio_arr)) {
        std::cerr << "Failed to read audio from disk" << std::endl;
        return 1;
    }

    original_audio_arr.resize(50000);

    // reconstruct audio
    if (!vocos_reconstruct_audio(vctx, original_audio_arr, params.n_threads)) {
        std::cerr << "Failed to reconstruct audio" << std::endl;
        return 1;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();
        const vocos_statistics * stats = vocos_get_statistics(vctx);

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, stats->t_load_us/1000.0f);
        printf("%s:     eval time = %8.2f ms\n", __func__, stats->t_compute_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    vocos_free(vctx);

    return 0;
}