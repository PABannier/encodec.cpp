#include "encodec.h"
#include "encoder.h"
#include "ggml.h"
#include "util.h"

#include <cmath>
#include <stdexcept>
#include <fstream>
#include <map>
#include <string>
#include <vector>

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

static void encodec_model_eval(
        std::vector<float>& raw_audio,
        encodec_model& model,
        int n_threads) {
    static size_t buf_size = 512u*MB;
    static void * buf      = malloc(buf_size);

    struct ggml_init_params params = { buf_size, buf, false };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph    gf   = {};
    gf.n_threads = n_threads;

    struct ggml_tensor * inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, raw_audio.size());
    memcpy(inp->data, raw_audio.data(), raw_audio.size()*ggml_element_size(inp));

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

    ggml_build_forward_expand(&gf, out);
    ggml_graph_compute       (ctx0, &gf);

    printf("\n");
    printf("seq_length   = %d\n", out->ne[0]);
    printf("n_channels   = %d\n", out->ne[1]);
    printf("out_channels = %d\n", out->ne[2]);
    printf("\n");

    out = ggml_view_2d(ctx0, out, 1000, out->ne[1], out->nb[1], 0);

    for(int i = 0; i < out->ne[1]; i++) {
        for (int j = 0; j < out->ne[0]; j++) {
            float val =  *(float *) ((char *) out->data + j*out->nb[0] + i*out->nb[1]);
            printf("%.4f ", val);
        }
        printf("\n");
    }

    // for(int i = 0; i < out->ne[1]; i++) {
    //     for (int j = 0; j < out->ne[0]; j++) {
    //         int32_t val =  *(int32_t *) ((char *) out->data + j*out->nb[0] + i*out->nb[1]);
    //         printf("%d ", val);
    //     }
    //     printf("\n");
    // }

    ggml_free(ctx0);
}

int main(int argc, char* argv[]) {
    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_load_us  = 0;
    int64_t t_compr_us = 0;

    encodec_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        std::string model_path = "./ggml_weights/ggml-model.bin";

        if (!encodec_model_load(model_path, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, model_path.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    // generate toy data
    std::vector<float> raw_audio(1000, 0.4);

    // encode
    const int64_t t_compr_us_start = ggml_time_us();

    // TODO change n_threads to be passed by params
    encodec_model_eval(raw_audio, model, 1);

    t_compr_us = ggml_time_us() - t_compr_us_start;

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s: compress time = %8.2f ms\n", __func__, t_compr_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}