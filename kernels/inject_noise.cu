#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "kernels/inject_noise.cuh"

namespace cg = cooperative_groups;

__global__ void get_opacity_sigmoid(
    const float* __restrict__ opacity,
    float* __restrict__ opacity_sigmoid,
    int num) {
    const float k = 100.0f;
    const float x0 = 0.995f;
    int idx = cg::this_grid().thread_rank();
    // Opacity sigmoid function: 1 / (1 + exp(-k * (x - x0)))
    // auto op_sigmoid = 1.0f / (1.0f + torch::exp(-k * ((1.0f - opacities) - x0)));
    if (idx < num) {
        opacity_sigmoid[idx] = 1.0f / (1.0f + exp(-k * ((1.0f - opacity[idx]) - x0)));
    }
}

void launch_get_opacity_sigmoid(
    const at::Tensor& opacity,
    at::Tensor& opacity_sigmoid,
    int num) {
    const int block_size = 256;
    const int grid_size = (num + block_size - 1) / block_size;
    auto stream = at::cuda::getCurrentCUDAStream();
    get_opacity_sigmoid<<<grid_size, block_size, 0, stream>>>(
        opacity.contiguous().const_data_ptr<float>(),
        opacity_sigmoid.contiguous().mutable_data_ptr<float>(),
        num);
}

__global__ void sgemv_3x3(const float* __restrict__ A, // [N, 3, 3]
                          const float* __restrict__ B, // [N, 3]
                          float* __restrict__ C,       // [N, 3]
                          int num) {
    int idx = cg::this_grid().thread_rank();
    if (idx < num) {
        float c[3] = {0.0f, 0.0f, 0.0f};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                c[i] += A[idx * 9 + i * 3 + j] * B[idx * 3 + j];
            }
        }
        for (int i = 0; i < 3; i++) {
            C[idx * 3 + i] += c[i];
        }
    }
}

void launch_sgemv_3x3(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C) {
    int num = A.size(0);
    TORCH_CHECK(A.dim() == 3 &&
                    B.dim() == 2 &&
                    C.dim() == 2 &&
                    B.size(0) == num &&
                    C.size(0) == num &&
                    A.size(1) == 3 &&
                    A.size(2) == 3 &&
                    B.size(1) == 3 &&
                    C.size(1) == 3,
                "sgemv_3x3: A B, and C must be [N, 3, 3], [N, 3] and [N, 3]");
    const int block_size = 256;
    const int grid_size = (num + block_size - 1) / block_size;
    auto stream = at::cuda::getCurrentCUDAStream();
    sgemv_3x3<<<grid_size, block_size, 0, stream>>>(
        A.contiguous().const_data_ptr<float>(),
        B.contiguous().const_data_ptr<float>(),
        C.contiguous().mutable_data_ptr<float>(),
        num);
}
