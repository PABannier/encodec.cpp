#include "ggml.h"

#include <random>
#include <vector>

static const size_t MB = 4*1024*1024;

static float randf() {
    return (float)(rand()) / (float)(rand());
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

struct ggml_tensor * forward_pass_lstm(
    struct ggml_context * ctx0,
    struct ggml_tensor * inp,
    struct ggml_tensor * weight_ih_l0,
    struct ggml_tensor * weight_hh_l0,
    struct ggml_tensor * bias_ih_l0,
    struct ggml_tensor * bias_hh_l0,
    struct ggml_tensor * weight_ih_l1,
    struct ggml_tensor * weight_hh_l1,
    struct ggml_tensor * bias_ih_l1,
    struct ggml_tensor * bias_hh_l1) {

    struct ggml_tensor * hs1 = forward_pass_lstm_unilayer(ctx0, inp, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0);
    struct ggml_tensor * out = forward_pass_lstm_unilayer(ctx0, hs1, weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1);

    return out;
}

int main() {
    struct ggml_init_params params = { 4*MB, NULL, false };
    struct ggml_context * ctx = ggml_init(params);

    const int seq_length  = 10;
    const int input_size  = 4;
    const int hidden_size = 7;

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_length, input_size);

    struct ggml_tensor * weight_ih_l0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_size , 4*hidden_size);
    struct ggml_tensor * weight_hh_l0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 4*hidden_size);
    struct ggml_tensor * weight_ih_l1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 4*hidden_size);
    struct ggml_tensor * weight_hh_l1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, 4*hidden_size);

    struct ggml_tensor * bias_ih_l0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*hidden_size);
    struct ggml_tensor * bias_hh_l0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*hidden_size);
    struct ggml_tensor * bias_ih_l1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*hidden_size);
    struct ggml_tensor * bias_hh_l1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*hidden_size);

    struct ggml_tensor * ans = forward_pass_lstm(ctx, inp, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1);

    struct ggml_cgraph gf = ggml_build_forward(ans);
    gf.n_threads = 1;

    ggml_set_f32(inp, 0.4f);

    ggml_set_f32(weight_ih_l0, 0.2f);
    ggml_set_f32(weight_hh_l0, -0.1f);
    ggml_set_f32(weight_ih_l1, 0.15f);
    ggml_set_f32(weight_hh_l1, -0.17f);

    ggml_set_f32(bias_ih_l0, 0.1f);
    ggml_set_f32(bias_hh_l0, -0.2f);
    ggml_set_f32(bias_ih_l1, 0.09f);
    ggml_set_f32(bias_hh_l1, -0.14f);

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
