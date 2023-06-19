#include "ggml.h"

#include <random>
#include <vector>

static const size_t MB = 4*1024*1024;

static float randf() {
    return (float)(rand()) / (float)(rand());
}

int main() {
    struct ggml_init_params params = { 4*MB, NULL, false };
    struct ggml_context * ctx = ggml_init(params);

    int n_channels     = 2;
    int n_out_channels = 2;
    int seq_length     = 3;
    int kernel_size    = 2;

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_length, n_channels);
    struct ggml_tensor * ker = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, n_out_channels, n_channels);

    struct ggml_tensor * ans = ggml_transpose_conv_1d(ctx, ker, inp, 1);

    struct ggml_cgraph gf = ggml_build_forward(ans);
    gf.n_threads = 1;

    std::vector<float> raw_data = {1, 2, 3, 4, 5, 6, 7, 8};
    memcpy(ker->data, raw_data.data(), raw_data.size()*ggml_element_size(ker));

    std::vector<float>raw_inp_data = {0, 2, -1, 1, 3, -2};
    memcpy(inp->data, raw_inp_data.data(), raw_inp_data.size()*ggml_element_size(inp));

    ggml_graph_compute(ctx, &gf);

    printf("inp=\n");
    for(int i = 0; i < inp->ne[1]; i++) {
        for (int j = 0; j < inp->ne[0]; j++) {
            float val =  *(float *) ((char *) inp->data + j*inp->nb[0] + i*inp->nb[1]);
            printf("%.4f ", val);
        }
        printf("\n");
    }

    printf("\n");
    
    printf("ker=\n");
    for(int i = 0; i < ker->ne[2]; i++) {
        for(int j = 0; j < ker->ne[1]; j++) {
            for (int k = 0; k < ker->ne[0]; k++) {
                float val =  *(float *) ((char *) ker->data + k*ker->nb[0] + j*ker->nb[1] + i*ker->nb[2]);
                printf("%.4f ", val);
            }
            printf("\n");
        }
    }

    printf("\n");

    printf("ans=\n");
    for(int i = 0; i < ans->ne[1]; i++) {
        for (int j = 0; j < ans->ne[0]; j++) {
            float val =  *(float *) ((char *) ans->data + j*ans->nb[0] + i*ans->nb[1]);
            printf("%.4f ", val);
        }
        printf("\n");
    }

    printf("\n");
    return 0;
}

// int main() {

//     struct ggml_init_params params = { 4*MB, NULL, false };
//     struct ggml_context * ctx = ggml_init(params);

//     int seq_length = 2;
//     int n_bins = 3;

//     struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_length, n_bins);
//     struct ggml_tensor * ans = ggml_argmax(ctx, inp);

//     struct ggml_cgraph gf = ggml_build_forward(ans);
//     gf.n_threads = 1;

//     std::vector<float> raw_data(n_bins * seq_length);
//     for (int i = 0; i < seq_length; i++) {
//         for (int j = 0; j < n_bins; j++) {
//                 raw_data[n_bins * i + j] = randf();
//         }
//     }

//     memcpy(inp->data, raw_data.data(), n_bins*seq_length*sizeof(float));

//     ggml_graph_compute(ctx, &gf);

//     printf("inp=\n");
//     for(int i = 0; i < inp->ne[1]; i++) {
//         for (int j = 0; j < inp->ne[0]; j++) {
//             float val =  *(float *) ((char *) inp->data + j*inp->nb[0] + i*inp->nb[1]);
//             printf("%.4f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");

//     printf("ans=\n");
//     for(int i = 0; i < ans->ne[1]; i++) {
//         for (int j = 0; j < ans->ne[0]; j++) {
//             float val =  *(float *) ((char *) ans->data + j*ans->nb[0] + i*ans->nb[1]);
//             printf("%.4f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");


//     return 0;
// }

// static struct ggml_tensor * quantizer_encode(
//                 ggml_context * ctx0,
//                  ggml_tensor * inp,
//                  ggml_tensor * embed,
//                  const int     n_q) {

//     const int seq_length = inp->ne[0];

//     struct ggml_tensor * codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, seq_length, n_q);

//     inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

//     struct ggml_tensor * residual = inp;

//     // seq_length
//     struct ggml_tensor * indices;

//     for (int i = 0; i < 1; i++) {
//         // compute distance
//         // [seq_length, n_bins]
//         struct ggml_tensor * dp = ggml_scale(
//                 ctx0, ggml_mul_mat(ctx0, embed, residual), ggml_new_f32(ctx0, -2.0f));

//         // [n_bins]
//         struct ggml_tensor * sqr_embed     = ggml_sqr(ctx0, embed);
//         struct ggml_tensor * sqr_embed_nrm = ggml_sum_rows(ctx0, sqr_embed);

//         // [seq_length]
//         struct ggml_tensor * sqr_inp     = ggml_sqr(ctx0, residual);
//         struct ggml_tensor * sqr_inp_nrm = ggml_sum_rows(ctx0, sqr_inp);

//         // [seq_length, n_bins]
//         struct ggml_tensor * dist = ggml_sub(ctx0, ggml_repeat(ctx0, sqr_inp_nrm, dp), dp);
//         dist = ggml_add(ctx0, ggml_repeat(ctx0, ggml_transpose(ctx0, sqr_embed_nrm), dist), dist);
//         dist = ggml_scale(ctx0, dist, ggml_new_f32(ctx0, -1.0f));

//         // take the argmax over the column dimension
//         // [seq_length]
//         indices = ggml_argmax(ctx0, dist);
//         indices = ggml_transpose(ctx0, indices);

//         // look up in embedding table
//         struct ggml_tensor * quantized = ggml_get_rows(ctx0, embed, indices);

//         residual = ggml_sub(ctx0, residual, quantized);

//         ggml_set_1d(ctx0, codes, indices, i*seq_length*ggml_element_size(codes));
//     }

//     // return codes;
//     return codes;
// }

// int main() {
//     struct ggml_init_params params = { 4*MB, NULL, false };
//     struct ggml_context * ctx = ggml_init(params);

//     const int seq_length = 5;
//     const int input_dim  = 4;
//     const int n_bins     = 3;
//     const int n_q        = 2;

//     struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_length, input_dim);
//     struct ggml_tensor * embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, n_bins);

//     struct ggml_tensor * ans = quantizer_encode(ctx, inp, embed, n_q);

//     struct ggml_cgraph gf = ggml_build_forward(ans);
//     gf.n_threads = 1;

//     ggml_set_f32(inp, 0.4f);
//     // ggml_set_f32(embed, -0.2f);

//     for (int i = 0; i < embed->ne[1]; i++) {
//         for (int j = 0; j < embed->ne[0]; j++) {
//             *(float *) ((char *) embed->data + j*embed->nb[0] + i*embed->nb[1]) = randf();
//         }
//     }

//     ggml_graph_compute(ctx, &gf);

//     printf("inp=\n");
//     for(int i = 0; i < inp->ne[1]; i++) {
//         for (int j = 0; j < inp->ne[0]; j++) {
//             float val =  *(float *) ((char *) inp->data + j*inp->nb[0] + i*inp->nb[1]);
//             printf("%.4f ", val);
//         }
//         printf("\n");
//     }

//     printf("embed=\n");
//     for(int i = 0; i < embed->ne[1]; i++) {
//         for (int j = 0; j < embed->ne[0]; j++) {
//             float val =  *(float *) ((char *) embed->data + j*embed->nb[0] + i*embed->nb[1]);
//             printf("%.4f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");

//     printf("ans=\n");
//     for(int i = 0; i < ans->ne[1]; i++) {
//         for (int j = 0; j < ans->ne[0]; j++) {
//             float val =  *(float *) ((char *) ans->data + j*ans->nb[0] + i*ans->nb[1]);
//             printf("%.4f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");
//     return 0;

// }


// int main() {
//     struct ggml_init_params params = { 4*MB, NULL, false };
//     struct ggml_context * ctx = ggml_init(params);

//     const int seq_length  = 10;
//     const int input_size  = 4;
//     const int hidden_size = 7;

//     struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_length, input_size);

//     struct ggml_tensor * weight_ih_l0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size , 4*hidden_size);
//     struct ggml_tensor * weight_hh_l0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 4*hidden_size);
//     struct ggml_tensor * weight_ih_l1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 4*hidden_size);
//     struct ggml_tensor * weight_hh_l1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 4*hidden_size);

//     struct ggml_tensor * bias_ih_l0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*hidden_size);
//     struct ggml_tensor * bias_hh_l0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*hidden_size);
//     struct ggml_tensor * bias_ih_l1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*hidden_size);
//     struct ggml_tensor * bias_hh_l1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*hidden_size);

//     struct ggml_tensor * ans = forward_pass_lstm(ctx, inp, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1);

//     struct ggml_cgraph gf = ggml_build_forward(ans);
//     gf.n_threads = 1;

//     ggml_set_f32(inp, 0.4f);

//     ggml_set_f32(weight_ih_l0, 0.2f);
//     ggml_set_f32(weight_hh_l0, -0.1f);
//     ggml_set_f32(weight_ih_l1, 0.15f);
//     ggml_set_f32(weight_hh_l1, -0.17f);

//     ggml_set_f32(bias_ih_l0, 0.1f);
//     ggml_set_f32(bias_hh_l0, -0.2f);
//     ggml_set_f32(bias_ih_l1, 0.09f);
//     ggml_set_f32(bias_hh_l1, -0.14f);

//     ggml_graph_compute(ctx, &gf);

//     printf("inp=\n");
//     for(int i = 0; i < inp->ne[1]; i++) {
//         for (int j = 0; j < inp->ne[0]; j++) {
//             float val =  *(float *) ((char *) inp->data + j*inp->nb[0] + i*inp->nb[1]);
//             printf("%.4f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");

//     printf("ans=\n");
//     for(int i = 0; i < ans->ne[1]; i++) {
//         for (int j = 0; j < ans->ne[0]; j++) {
//             float val =  *(float *) ((char *) ans->data + j*ans->nb[0] + i*ans->nb[1]);
//             printf("%.4f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");
//     return 0;

// }

// int main() {
//     struct ggml_init_params params = { 4*MB, NULL, false };
//     struct ggml_context * ctx = ggml_init(params);

//     int n_channels = 4;
//     int n_out_channels = 5;
//     int seq_length = 12;
//     int kernel_size = 4;

//     struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_length, n_channels);
//     struct ggml_tensor * ker = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, n_channels, n_out_channels);

//     struct ggml_tensor * ans = ggml_conv_1d_8s(ctx, ker, inp);

//     struct ggml_cgraph gf = ggml_build_forward(ans);
//     gf.n_threads = 1;

//     std::vector<float> raw_data(ggml_nelements(ker));
//     for (size_t i = 0; i < n_channels; i++) {
//         for (size_t j = 0; j < kernel_size; j++) {
//             for (size_t k = 0; k < n_out_channels; k++) {
//                 raw_data[n_out_channels*kernel_size*i + n_out_channels*j + k] = randf();
//             }
//         }
//     }
//     memcpy(ker->data, raw_data.data(), raw_data.size()*ggml_element_size(ker));

//     std::vector<float>raw_inp_data(ggml_nelements(inp));
//     for (size_t i = 0; i < n_channels; i++) {
//         for (size_t j = 0; j < seq_length; j++) {
//             raw_inp_data[seq_length*i + j] = 0.4f;
//         }
//     }
//     memcpy(inp->data, raw_inp_data.data(), raw_inp_data.size()*ggml_element_size(inp));

//     ggml_graph_compute(ctx, &gf);

//     printf("inp=\n");
//     for(int i = 0; i < inp->ne[1]; i++) {
//         for (int j = 0; j < inp->ne[0]; j++) {
//             float val =  *(float *) ((char *) inp->data + j*inp->nb[0] + i*inp->nb[1]);
//             printf("%.4f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");

//     printf("ans=\n");
//     for(int i = 0; i < ans->ne[1]; i++) {
//         for (int j = 0; j < ans->ne[0]; j++) {
//             float val =  *(float *) ((char *) ans->data + j*ans->nb[0] + i*ans->nb[1]);
//             printf("%.4f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");
//     return 0;
// }

// int main() {
//     struct ggml_init_params params = { 4*MB, NULL, false };
//     struct ggml_context * ctx = ggml_init(params);

//     int n_channels = 4;
//     int seq_length = 8;

//     struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_length, n_channels);

//     int padding[2] = {2, 0};
//     struct ggml_tensor * ans = ggml_pad_1d_reflective(ctx, inp, padding);

//     struct ggml_cgraph gf = ggml_build_forward(ans);
//     gf.n_threads = 1;

//     std::vector<float> raw_data(ggml_nelements(inp));
//     for (size_t i = 0; i < n_channels; i++) {
//         // float v = randf();
//         for (size_t j = 0; j < seq_length; j++) {
//             // raw_data[seq_length*i + j] = v;
//             raw_data[seq_length*i + j] = randf();
//         }
//     }
//     memcpy(inp->data, raw_data.data(), raw_data.size()*ggml_element_size(inp));

//     ggml_graph_compute(ctx, &gf);

//     printf("inp=\n");
//     for(int i = 0; i < inp->ne[1]; i++) {
//         for (int j = 0; j < inp->ne[0]; j++) {
//             float val =  *(float *) ((char *) inp->data + j*inp->nb[0] + i*inp->nb[1]);
//             printf("%.4f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");

//     printf("ans=\n");
//     for(int i = 0; i < ans->ne[1]; i++) {
//         for (int j = 0; j < ans->ne[0]; j++) {
//             float val =  *(float *) ((char *) ans->data + j*ans->nb[0] + i*ans->nb[1]);
//             printf("%.4f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");
//     return 0;
// }

// int main() {
//      struct ggml_init_params params = { 4*MB, NULL, false };
//     struct ggml_context * ctx = ggml_init(params);

//     struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, 7);
//     struct ggml_tensor * k   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 7, 14);

//     int padding[2] = {2, 0};
//     // struct ggml_tensor * padded_inp = ggml_pad_1d_constant(ctx, inp, padding, 0);
//     struct ggml_tensor * padded_inp = ggml_pad_1d_reflective(ctx, inp, padding);
//     struct ggml_tensor * ans = ggml_conv_1d_1s(ctx, k, padded_inp);

//     struct ggml_cgraph gf = ggml_build_forward(ans);
//     gf.n_threads = 1;

//     std::vector<float> raw_data1(ggml_nelements(inp), 0.4);
//     memcpy(inp->data, raw_data1.data(), raw_data1.size()*ggml_element_size(inp));

//     std::vector<float> raw_data2(ggml_nelements(k));
//     for (size_t i = 0; i < raw_data2.size(); i++) { raw_data2[i] = randf(); }
//     memcpy(k->data, raw_data2.data(), raw_data2.size()*ggml_element_size(k));

//     ggml_graph_compute(ctx, &gf);

//     printf("ans=\n");
//     for(int i = 0; i < ans->ne[1]; i++) {
//         for (int j = 0; j < ans->ne[0]; j++) {
//             float val =  *(float *) ((char *) ans->data + j*ans->nb[0] + i*ans->nb[1]);
//             printf("%.4f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");
//     return 0;
// }

// int main() {
//      struct ggml_init_params params = { 4*MB, NULL, false };
//     struct ggml_context * ctx = ggml_init(params);

//     struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);

//     int padding[2] = {2, 2};
//     struct ggml_tensor * ans = ggml_pad_1d_reflective(ctx, inp, padding);

//     struct ggml_cgraph gf = ggml_build_forward(ans);

//     std::vector<float> raw_data1 = {0.1, 0.2, 0.3};
//     memcpy(inp->data, raw_data1.data(), raw_data1.size()*ggml_element_size(inp));

//     std::vector<float> raw_data2 = {0.4, 0.5, 0.6};
//     memcpy((float *) ((char *) inp->data + 3*4), raw_data2.data(), raw_data2.size()*ggml_element_size(inp));

//     ggml_graph_compute(ctx, &gf);

//     printf("inp=\n");
//     for(int i = 0; i < inp->ne[0]; i++) {
//         for (int j = 0; j < inp->ne[1]; j++) {
//             float val =  *(float *) ((char *) inp->data + j*inp->nb[1] + i*inp->nb[0]);
//             printf("%.2f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");

//     printf("ans=\n");
//     for(int i = 0; i < ans->ne[0]; i++) {
//         for (int j = 0; j < ans->ne[1]; j++) {
//             float val =  *(float *) ((char *) ans->data + j*ans->nb[1] + i*ans->nb[0]);
//             printf("%.2f ", val);
//         }
//         printf("\n");
//     }

//     printf("\n");
//     return 0;
// }
