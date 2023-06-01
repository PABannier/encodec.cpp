#include "ggml.h"

#include <vector>

static const size_t MB = 4*1024*1024;

int main() {
     struct ggml_init_params params = { 4*MB, NULL, false };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);

    int padding[2] = {2, 2};
    struct ggml_tensor * ans = ggml_pad_1d_reflective(ctx, inp, padding);

    struct ggml_cgraph gf = ggml_build_forward(ans);

    std::vector<float> raw_data1 = {0.1, 0.2, 0.3};
    memcpy(inp->data, raw_data1.data(), raw_data1.size()*ggml_element_size(inp));

    std::vector<float> raw_data2 = {0.4, 0.5, 0.6};
    memcpy((float *) ((char *) inp->data + 3*4), raw_data2.data(), raw_data2.size()*ggml_element_size(inp));

    ggml_graph_compute(ctx, &gf);

    printf("inp=\n");
    for(int i = 0; i < inp->ne[0]; i++) {
        for (int j = 0; j < inp->ne[1]; j++) {
            float val =  *(float *) ((char *) inp->data + j*inp->nb[1] + i*inp->nb[0]);
            printf("%.2f ", val);
        }
        printf("\n");
    }

    printf("\n");

    printf("ans=\n");
    for(int i = 0; i < ans->ne[0]; i++) {
        for (int j = 0; j < ans->ne[1]; j++) {
            float val =  *(float *) ((char *) ans->data + j*ans->nb[1] + i*ans->nb[0]);
            printf("%.2f ", val);
        }
        printf("\n");
    }

    printf("\n");
    return 0;
}