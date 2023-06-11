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

    int n_channels = 4;
    int n_out_channels = 5;
    int seq_length = 8;
    int kernel_size = 3;

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_length, n_channels);
    struct ggml_tensor * ker = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, n_channels, n_out_channels);

    struct ggml_tensor * ans = ggml_conv_1d_2s(ctx, ker, inp);

    struct ggml_cgraph gf = ggml_build_forward(ans);
    gf.n_threads = 1;

    std::vector<float> raw_data(ggml_nelements(ker));
    for (size_t i = 0; i < n_channels; i++) {
        for (size_t j = 0; j < kernel_size; j++) {
            for (size_t k = 0; k < n_out_channels; k++) {
                raw_data[n_out_channels*kernel_size*i + n_out_channels*j + k] = randf();
            }
        }
    }
    memcpy(ker->data, raw_data.data(), raw_data.size()*ggml_element_size(ker));

    std::vector<float>raw_inp_data(ggml_nelements(inp));
    for (size_t i = 0; i < n_channels; i++) {
        for (size_t j = 0; j < seq_length; j++) {
            raw_inp_data[seq_length*i + j] = 0.4f;
        }
    }
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
