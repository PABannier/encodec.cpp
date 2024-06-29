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

#define VOCOS_FILE_MAGIC 'ggml'

static const size_t MB = 1024 * 1024;

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
    struct ggml_tensor * features = NULL;
    struct ggml_tensor * codes    = NULL;
    struct ggml_tensor * decoded  = NULL;

    std::vector<int32_t> out_codes;
    std::vector<float>   out_audio;

    // statistics
    struct vocos_statistics stats;
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

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    struct vocos_context *vctx = vocos_load_model(model_path);
    if (!vctx) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    vocos_free(vctx);

    return 0;
}