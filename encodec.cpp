#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml/src/ggml-impl.h"

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

#include "decoder.h"
#include "encoder.h"
#include "lstm.h"
#include "ops.h"
#include "utils.h"
#include "quantizer.h"

#define ENCODEC_FILE_MAGIC 'ggml'
#define ENCODEC_MAX_NODES 80000

typedef enum {
    // Run the end-to-end encoder-decoder pipeline
    FULL = 0,
    // Encode an audio (encoder + quantizer encode)
    ENCODE = 1,
    // Decode an audio from a compressed representation (quantizer decode + decoder)
    DECODE = 2,
} encodec_run_mode_t;

struct encodec_hparams {
    // The number of input channels is always 1 (mono).
    int32_t in_channels = 1;
    // The hidden dimension for the codebook.
    int32_t hidden_dim = 128;
    // The number of filters for the first convolution.
    int32_t n_filters = 32;
    // The filter size for upsampling and downsampling.
    int32_t ratios[4] = {8, 5, 4, 2};
    // The kernel size for the first convolution.
    int32_t kernel_size = 7;
    // The kernel size for the residual blocks.
    int32_t residual_kernel_size = 3;
    // Compression
    int32_t compress = 2;
    // The number of layers in the LSTM modules.
    int32_t n_lstm_layers = 2;
    // The stride of the first convolution.
    int32_t stride = 1;

    // The dimension of the codebook.
    int32_t n_bins = 1024;
    // The sample rate of the model.
    int32_t sr = 24000;
    // The bandwidth.
    int32_t bandwidth = 24;

    // The number of codebooks.
    int32_t n_q = 32;
    // The product of the ratios.
    int32_t hop_length = 1;

    // File type of model weights.
    int32_t ftype;
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

struct encodec_ggml_cgraph_deleter {
    void operator()(struct ggml_cgraph * cgraph) {
        if (cgraph->nodes)
            free(cgraph->nodes);
        if (cgraph->leafs)
            free(cgraph->leafs);
        if (cgraph->visited_hash_set.keys)
            free(cgraph->visited_hash_set.keys);
        if (cgraph->grads)
            free(cgraph->grads);
        free(cgraph);
    }
};

struct encodec_context {
    encodec_model model;

    // computational graph stored on the heap to avoid stack overflows
    // the computational graph grows with the sequence length (because of the LSTM)
    // which requires a lot of nodes
    std::unique_ptr<struct ggml_cgraph, encodec_ggml_cgraph_deleter> gf;

    // buffer for model evaluation
    ggml_backend_buffer_t buf_compute;

    // tensor graph allocator
    ggml_gallocr_t allocr = NULL;

    // intermediate steps
    struct ggml_tensor *encoded = NULL;  // Encoded audio
    struct ggml_tensor *codes = NULL;    // Quantized representation of audio in codebook
    struct ggml_tensor *decoded = NULL;  // Reconstructed audio from codes

    std::vector<int32_t> out_codes;
    std::vector<float> out_audio;

    // statistics
    encodec_statistics stats;
};

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

    // create the ggml context
    {
        size_t n_tensors = ((4 * 2) * 4 + 2 + 4 * model.hparams.n_lstm_layers + 2) * 2;  // encoder and decoder
        n_tensors += model.hparams.n_q * 1;                                              // quantizer
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

    // create the tensors for the model
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

    // allocate the model tensors in a backend buffer
    model.buffer_w = ggml_backend_alloc_ctx_tensors(ctx, model.backend);

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

            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        printf("%s: model size = %.2f MB\n", __func__, total_size / 1024.0 / 1024.0);
    }

    infile.close();

    return true;
}

// Create a new ggml_cgraph with the given size (usually ENCODEC_MAX_NODES). We need a
// custom function since the graph is so large, it overpasses the max built-in ggml
// default size.
static struct ggml_cgraph * encodec_ggml_cgraph_create(size_t size) {
    struct ggml_cgraph * cgraph = (struct ggml_cgraph *)calloc(1, sizeof(struct ggml_cgraph));
    cgraph->size = size;
    cgraph->n_nodes = 0;
    cgraph->n_leafs = 0;
    cgraph->nodes = (struct ggml_tensor **)calloc(1, size * sizeof(struct ggml_tensor *));
    cgraph->leafs = (struct ggml_tensor **)calloc(1, size * sizeof(struct ggml_tensor *));

    // next primes after powers of two
    static const size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    static const size_t n_primes = sizeof(primes)/sizeof(primes[0]);

    // find the smallest prime that is larger or equal to size
    size_t l = 0;
    size_t r = n_primes;
    while (l < r) {
        size_t m = (l + r)/2;
        if (primes[m] < size * 2) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    size_t hash_size = l < n_primes ? primes[l] : (size * 2 + 1);

    cgraph->visited_hash_set.size = hash_size;
    cgraph->visited_hash_set.keys = (struct ggml_tensor **)calloc(1, hash_size * sizeof(struct ggml_tensor *));
    cgraph->visited_hash_set.used = (ggml_bitset_t *)calloc(1, ggml_bitset_size(hash_size) * sizeof(ggml_bitset_t));
    cgraph->order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT;

    return cgraph;
}

void encodec_build_graph(struct encodec_context *ectx,
                         const float * inp_audio,
                         const int n_samples,
                         const encodec_run_mode_t mode) {
    assert(mode == encodec_run_mode_t::FULL || mode == encodec_run_mode_t::ENCODE);

    const auto & model   = ectx->model;
    const auto & hparams = model.hparams;
    const auto & allocr  = ectx->allocr;

    auto & gf = ectx->gf;

    const int *ratios       = hparams.ratios;
    const int kernel_size   = hparams.kernel_size;
    const int res_kernel_sz = hparams.residual_kernel_size;
    const int stride        = hparams.stride;
    const int n_bins        = hparams.n_bins;
    const int n_q           = hparams.n_q;
    const int sr            = hparams.sr;
    const int bandwidth     = hparams.bandwidth;
    const int hop_length    = hparams.hop_length;
    const int hidden_dim    = hparams.hidden_dim;

    // since we are using ggml-alloc, this buffer only needs enough space to hold the
    // ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead() * ENCODEC_MAX_NODES + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,  // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    struct ggml_context *ctx0 = ggml_init(ggml_params);

    gf = std::unique_ptr<struct ggml_cgraph, encodec_ggml_cgraph_deleter>(encodec_ggml_cgraph_create(ENCODEC_MAX_NODES));

    struct ggml_tensor *inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_samples);
    ggml_set_name(inp, "inp");
    ggml_set_input(inp);

    const struct encodec_encoder   * encoder   = &model.encoder;
    const struct encodec_quantizer * quantizer = &model.quantizer;
    const struct encodec_decoder   * decoder   = &model.decoder;

    struct ggml_tensor * encoded = encodec_forward_encoder(
        encoder, ctx0, inp, ratios, kernel_size, res_kernel_sz, stride);

    struct ggml_tensor * codes = encodec_forward_quantizer_encode(
        quantizer, ctx0, encoded, n_bins, sr, bandwidth, hop_length);

    struct ggml_tensor * quantized = encodec_forward_quantizer_decode(
        quantizer, ctx0, codes, hidden_dim, n_bins, sr, bandwidth, hop_length);

    struct ggml_tensor * decoded = encodec_forward_decoder(
        decoder, ctx0, quantized, ratios, kernel_size, res_kernel_sz, stride);

    switch (mode) {
        case encodec_run_mode_t::FULL: {
            ggml_set_name(decoded, "decoded");
            ggml_set_output(decoded);
            ggml_build_forward_expand(gf.get(), decoded);
        } break;
        case encodec_run_mode_t::ENCODE: {
            ggml_set_name(codes, "codes");
            ggml_set_output(codes);
            ggml_build_forward_expand(gf.get(), codes);
        } break;
        case encodec_run_mode_t::DECODE: {
            assert(false);
        } break;
        default: {
            fprintf(stderr, "%s: unknown run mode\n", __func__);
        } break;
    }

    ggml_free(ctx0);

    ectx->encoded = encoded;
    ectx->codes   = codes;
    ectx->decoded = decoded;
}

void encodec_build_graph(struct encodec_context *ectx, const int32_t *codes,
                         const int n_codes, const encodec_run_mode_t mode) {
    assert(mode == encodec_run_mode_t::DECODE);

    const auto & model   = ectx->model;
    const auto & hparams = model.hparams;
    const auto & allocr  = ectx->allocr;

    auto & gf = ectx->gf;

    const int n_bins        = hparams.n_bins;
    const int sr            = hparams.sr;
    const int bandwidth     = hparams.bandwidth;
    const int hop_length    = hparams.hop_length;
    const int hidden_dim    = hparams.hidden_dim;
    const int * ratios      = hparams.ratios;
    const int kernel_size   = hparams.kernel_size;
    const int res_kernel_sz = hparams.residual_kernel_size;
    const int stride        = hparams.stride;

    const int frame_rate = (int)ceilf(sr / hop_length);
    const int n_q = get_num_quantizers_for_bandwidth(n_bins, frame_rate, bandwidth);

    if (n_codes % n_q != 0) {
        fprintf(stderr, "%s: invalid number of codes\n", __func__);
        assert(false);
    }

    const int N = n_codes / n_q;

    static size_t buf_size = ggml_tensor_overhead() * ENCODEC_MAX_NODES + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context *ctx0 = ggml_init(ggml_params);

    gf = std::unique_ptr<struct ggml_cgraph, encodec_ggml_cgraph_deleter>(encodec_ggml_cgraph_create(ENCODEC_MAX_NODES));

    struct ggml_tensor *inp_codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, N, n_q);
    ggml_set_name(inp_codes, "inp_codes");
    ggml_set_input(inp_codes);

    const struct encodec_quantizer * quantizer = &model.quantizer;
    const struct encodec_decoder   * decoder   = &model.decoder;

    struct ggml_tensor *quantized = encodec_forward_quantizer_decode(
        quantizer, ctx0, inp_codes, hidden_dim, n_bins, sr, bandwidth, hop_length
    );

    struct ggml_tensor *decoded = encodec_forward_decoder(
        decoder, ctx0, quantized, ratios, kernel_size, res_kernel_sz, stride
    );

    switch (mode) {
        case encodec_run_mode_t::DECODE: {
            ggml_set_name(decoded, "decoded");
            ggml_set_output(decoded);
            ggml_build_forward_expand(gf.get(), decoded);
        } break;
        default: {
            fprintf(stderr, "%s: unknown run mode\n", __func__);
            assert(false);
        } break;
    }

    ggml_free(ctx0);

    ectx->codes   = inp_codes;
    ectx->decoded = decoded;
}

static void encodec_zero_tensor(struct ggml_cgraph *gf, const char *name) {
    struct ggml_tensor *tensor = ggml_graph_get_tensor(gf, name);
    ggml_set_zero(tensor);
}

bool encodec_eval_internal(struct encodec_context *ectx, const float * raw_audio,
                           const int n_samples, const int n_threads,
                           const encodec_run_mode_t mode) {
    assert(mode == encodec_run_mode_t::FULL || mode == encodec_run_mode_t::ENCODE);
    auto & model  = ectx->model;
    auto & allocr = ectx->allocr;
    auto & gf     = ectx->gf;

    encodec_build_graph(ectx, raw_audio, n_samples, mode);

    // allocate the graph tensors
    ggml_gallocr_alloc_graph(allocr, gf.get());

    // set the graph inputs
    struct ggml_tensor * inp = ggml_graph_get_tensor(gf.get(), "inp");
    ggml_backend_tensor_set(inp, raw_audio, 0, n_samples * ggml_element_size(inp));

    // make sure accumulation tensor are zeroed
    encodec_zero_tensor(gf.get(), "enc_l0_ht");
    encodec_zero_tensor(gf.get(), "enc_l1_ht");
    encodec_zero_tensor(gf.get(), "enc_l0_ct");
    encodec_zero_tensor(gf.get(), "enc_l1_ct");

    if (mode == encodec_run_mode_t::FULL) {
        encodec_zero_tensor(gf.get(), "dec_l0_ht");
        encodec_zero_tensor(gf.get(), "dec_l1_ht");
        encodec_zero_tensor(gf.get(), "dec_l0_ct");
        encodec_zero_tensor(gf.get(), "dec_l1_ct");

        encodec_zero_tensor(gf.get(), "quantized_out");
    }

    // run the computation
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf.get());

    return true;
}

bool encodec_eval_internal(struct encodec_context *ectx, const int32_t *codes,
                           const int n_codes, const int n_threads,
                           const encodec_run_mode_t mode) {
    assert(mode == encodec_run_mode_t::DECODE);

    auto & model  = ectx->model;
    auto & allocr = ectx->allocr;
    auto & gf     = ectx->gf;

    encodec_build_graph(ectx, codes, n_codes, mode);

    // allocate the graph tensors
    ggml_gallocr_alloc_graph(allocr, gf.get());

    // set the graph inputs
    struct ggml_tensor * inp = ggml_graph_get_tensor(gf.get(), "inp_codes");
    ggml_backend_tensor_set(inp, codes, 0, n_codes * ggml_element_size(inp));

    // make sure accumulation tensors are zeroed
    encodec_zero_tensor(gf.get(), "dec_l0_ht");
    encodec_zero_tensor(gf.get(), "dec_l1_ht");
    encodec_zero_tensor(gf.get(), "dec_l0_ct");
    encodec_zero_tensor(gf.get(), "dec_l1_ct");

    encodec_zero_tensor(gf.get(), "quantized_out");

    // run the computation
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf.get());

    return true;
}

bool encodec_eval(struct encodec_context *ectx, const float *raw_audio,
                  const int n_samples, const int n_threads,
                  const encodec_run_mode_t mode) {
    const int64_t t_start_us = ggml_time_us();

    // allocate the compute buffer
    {
        // create a graph allocator with the backend's default buffer type
        ectx->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ectx->model.backend));

        // create the graph for memory usage estimation
        encodec_build_graph(ectx, raw_audio, n_samples, mode);

        // pre-allocate the compute buffer
        ggml_gallocr_reserve(ectx->allocr, ectx->gf.get());
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

bool encodec_eval(struct encodec_context *ectx, const int32_t *codes,
                  const int n_codes, const int n_threads,
                  const encodec_run_mode_t mode) {
    const int64_t t_start_ms = ggml_time_us();

    // allocate the compute buffer
    {
        // create a graph allocator with the backend's default buffer type
        ectx->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ectx->model.backend));

        // create the graph for memory usage estimation
        encodec_build_graph(ectx, codes, n_codes, mode);

        // pre-allocate the compute buffer
        ggml_gallocr_reserve(ectx->allocr, ectx->gf.get());
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

bool encodec_reconstruct_audio(struct encodec_context *ectx, const float *raw_audio,
                                const int n_samples, const int n_threads) {
    if (raw_audio == nullptr) {
        fprintf(stderr, "%s: null input audio\n", __func__);
        return false;
    }

    if (!encodec_eval(ectx, raw_audio, n_samples, n_threads, encodec_run_mode_t::FULL)) {
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

bool encodec_compress_audio(struct encodec_context *ectx, const float *raw_audio,
                             const int n_samples, const int n_threads) {
    if (!encodec_eval(ectx, raw_audio, n_samples, n_threads, encodec_run_mode_t::ENCODE)) {
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

bool encodec_decompress_audio(struct encodec_context *ectx, const int32_t *codes,
                              const int n_codes, const int n_threads) {
    if (!encodec_eval(ectx, codes, n_codes, n_threads, encodec_run_mode_t::DECODE)) {
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
    ectx->model.hparams.n_q        = get_num_codebooks(bandwidth, hop_length, sr);

    ectx->stats.t_load_us          = ggml_time_us() - t_start_load_us;

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
