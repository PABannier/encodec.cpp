#include <fstream>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

void print_tensor(struct ggml_tensor * a) {
    float sum = 0;
    if (a) {
        for (int i = 0; i < a->ne[3]; i++) {
            for (int j = 0; j < a->ne[2]; j++) {
                for (int k = 0; k < a->ne[1]; k++) {
                    for (int l = 0; l < a->ne[0]; l++) {
                        if (a->type == GGML_TYPE_F32) {
                            float * aval = (float *) (
                                (char *) a->data + i*a->nb[3] + j*a->nb[2] + k*a->nb[1] + l*a->nb[0]);
                            sum += *aval;
                            printf("%.4f ", *aval);
                        } else if (a->type == GGML_TYPE_I32) {
                            int32_t * aval = (int32_t *) (
                                (char *) a->data + i*a->nb[3] + j*a->nb[2] + k*a->nb[1] + l*a->nb[0]);
                            sum += (float) *aval;
                            printf("%d ", *aval);
                        } else {
                            throw;
                        }
                    }
                    printf("\n");
                }
                printf("\n\n");
            }
        }
        printf("sum=%.2f\n", sum);
        printf("shape=[%d, %d, %d, %d]\n", a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
        printf("--\n");
    }
}

inline float ggml_get_f32(struct ggml_tensor * t, int i0, int i1, int i2, int i3) {
    float * addr = (float *) ((char *) t->data + i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0]);
    return *addr;
}

void printf_summary_statistics(struct ggml_tensor * t) {
    printf("%s:     ne = [%lld, %lld, %lld, %lld]\n", __func__, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    printf("%s:     type = %d\n", __func__, t->type);
    printf("\n");
}

void compare_tensors(struct ggml_tensor * t1, struct ggml_tensor * t2) {
    assert(t1->type == GGML_TYPE_F32);
    assert(t2->type == GGML_TYPE_F32);

    for (int i = 0; i < 4; i++) {
        if (t1->ne[i] != t2->ne[i]) {
            printf("mismatch in ne[%d]: %lld != %lld\n", i, t1->ne[i], t2->ne[i]);
            exit(1);
        }
    }

    for (int i3 = 0; i3 < t1->ne[3]; i3++) {
        for (int i2 = 0; i2 < t1->ne[2]; i2++) {
            for (int i1 = 0; i1 < t1->ne[1]; i1++) {
                for (int i0 = 0; i0 < t1->ne[0]; i0++) {
                    float v1 = ggml_get_f32(t1, i0, i1, i2, i3);
                    float v2 = ggml_get_f32(t2, i0, i1, i2, i3);
                    if (abs(v1 - v2) > 1e-1) {
                        printf("mismatch at [%d, %d, %d, %d]: %f != %f\n", i0, i1, i2, i3, v1, v2);
                        exit(1);
                    }
                }
            }
        }
    }
}

struct ggml_context * make_ctx(void) {
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 1024,  // 1GB
    };
    return ggml_init(params);
}

void read_data_into_tensor(std::ifstream & infile, struct ggml_tensor * t) {
    std::vector<char> read_buf;
    read_buf.resize(ggml_nbytes(t));
    infile.read(read_buf.data(), ggml_nbytes(t));
    memcpy(t->data, read_buf.data(), ggml_nbytes(t));
}

int run_test(std::string fname) {
    int32_t stride, ttype;

    int32_t ne0x, ne1x;
    int32_t ne0y, ne1y;
    int32_t ne0w, ne1w, ne2w;

    struct ggml_tensor * x;
    struct ggml_tensor * w;
    struct ggml_tensor * y;

    struct ggml_context * ctx = make_ctx();

    {
        // read tensors from dumped PyTorch example
        auto infile = std::ifstream(fname, std::ios::binary);
        if(!infile) {
            printf("failed to open %s\n", fname.c_str());
            exit(1);
        }

        infile.read((char *) &stride, sizeof(int32_t));
        infile.read((char *) &ttype, sizeof(int32_t));

        // x
        {
            infile.read((char *) &ne0x, sizeof(int32_t));
            infile.read((char *) &ne1x, sizeof(int32_t));

            x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0x, ne1x);
            read_data_into_tensor(infile, x);
            // printf_summary_statistics(x);
        }

        // w
        {
            infile.read((char *) &ne0w, sizeof(int32_t));
            infile.read((char *) &ne1w, sizeof(int32_t));
            infile.read((char *) &ne2w, sizeof(int32_t));

            if (ttype == 0) {
                w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0w, ne1w, ne2w);
            } else {
                w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, ne0w, ne1w, ne2w);
            }
            read_data_into_tensor(infile, w);
            // printf_summary_statistics(w);
        }

        // y
        {
            infile.read((char *) &ne0y, sizeof(int32_t));
            infile.read((char *) &ne1y, sizeof(int32_t));

            y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0y, ne1y);
            read_data_into_tensor(infile, y);
            // printf_summary_statistics(y);
        }

        infile.close();
    }

    // computation
    struct ggml_tensor * y_hat = ggml_conv_transpose_1d(ctx, w, x, stride /* s0 */, 0 /* p0 */, 1 /* d0 */);

    struct ggml_cgraph gf = ggml_build_forward(y_hat);

    ggml_graph_compute_with_ctx(ctx, &gf, 4 /* n_threads */);
    // print_tensor(y_hat);

    // check results
    compare_tensors(y, y_hat);

    printf("passed.\n");

    // working
    return 0;
}

int main(void) {
    // run_test("conv_transpose_easy.bin");
    // run_test("conv_transpose_medium.bin");
    // run_test("conv_transpose_hard.bin");
    // run_test("dumped_tensors.bin");

    run_test("conv_transpose_easy_f16.bin");
    run_test("conv_transpose_medium_f16.bin");
    run_test("conv_transpose_hard_f16.bin");
    run_test("dumped_tensors_f16.bin");
}
