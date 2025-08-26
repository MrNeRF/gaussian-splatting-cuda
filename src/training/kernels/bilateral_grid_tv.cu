#include "kernels/bilateral_grid.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gs {
    namespace bilateral_grid {

        // Forward pass - compute total variation loss
        __global__ void tv_loss_forward_kernel(
            const float* __restrict__ grids, // [N, 12, L, H, W]
            float* __restrict__ tv_loss,     // scalar output
            int N, int L, int H, int W) {
            // Use block reduction for efficiency
            typedef cub::BlockReduce<float, 256> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;

            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = gridDim.x * blockDim.x;
            int total = N * L * H * W;

            float local_sum = 0.0f;

            // Grid-stride loop
            for (int idx = tid; idx < total; idx += stride) {
                // Decode position
                int tmp = idx;
                int wi = tmp % W;
                tmp /= W;
                int hi = tmp % H;
                tmp /= H;
                int li = tmp % L;
                tmp /= L;
                int ni = tmp;

// Process all 12 channels
#pragma unroll 12
                for (int ci = 0; ci < 12; ci++) {
                    int base = (ni * 12 + ci) * L * H * W;
                    int cell_idx = base + (li * H + hi) * W + wi;

                    float val = grids[cell_idx];

                    // X-direction
                    if (wi > 0) {
                        float val0 = grids[cell_idx - 1];
                        float diff = val - val0;
                        local_sum += diff * diff / (L * H * (W - 1));
                    }

                    // Y-direction
                    if (hi > 0) {
                        float val0 = grids[cell_idx - W];
                        float diff = val - val0;
                        local_sum += diff * diff / (L * (H - 1) * W);
                    }

                    // Z-direction
                    if (li > 0) {
                        float val0 = grids[cell_idx - W * H];
                        float diff = val - val0;
                        local_sum += diff * diff / ((L - 1) * H * W);
                    }
                }
            }

            local_sum /= (12 * N);

            // Block-level reduction
            local_sum = BlockReduce(temp_storage).Sum(local_sum);

            // Only thread 0 writes the result
            if (threadIdx.x == 0) {
                atomicAdd(tv_loss, local_sum);
            }
        }

        // Backward pass - compute gradients of TV loss
        __global__ void tv_loss_backward_kernel(
            const float* __restrict__ grids, // [N, 12, L, H, W]
            const float grad_output,         // scalar gradient
            float* __restrict__ grad_grids,  // [N, 12, L, H, W]
            int N, int L, int H, int W) {
            const size_t total = (size_t)N * 12 * L * H * W;
            size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = gridDim.x * blockDim.x;

            // Scaling factors
            const float s = grad_output / (6 * N);
            const float sx = s / (float)(L * H * (W - 1));
            const float sy = s / (float)(L * (H - 1) * W);
            const float sz = s / (float)((L - 1) * H * W);

            // Grid-stride loop
            for (size_t cell_idx = tid; cell_idx < total; cell_idx += stride) {
                // Decode position
                size_t idx = cell_idx;
                const int wi = idx % W;
                idx /= W;
                const int hi = idx % H;
                idx /= H;
                const int li = idx % L;
                idx /= L;

                float half_grad = 0.0f;
                const float val = grids[cell_idx];

                // X-direction gradients
                if (wi > 0) {
                    float val0 = grids[cell_idx - 1];
                    half_grad += (val - val0) * sx;
                }
                if (wi < W - 1) {
                    float val0 = grids[cell_idx + 1];
                    half_grad += (val - val0) * sx;
                }

                // Y-direction gradients
                if (hi > 0) {
                    float val0 = grids[cell_idx - W];
                    half_grad += (val - val0) * sy;
                }
                if (hi < H - 1) {
                    float val0 = grids[cell_idx + W];
                    half_grad += (val - val0) * sy;
                }

                // Z-direction gradients
                if (li > 0) {
                    float val0 = grids[cell_idx - W * H];
                    half_grad += (val - val0) * sz;
                }
                if (li < L - 1) {
                    float val0 = grids[cell_idx + W * H];
                    half_grad += (val - val0) * sz;
                }

                grad_grids[cell_idx] = half_grad;
            }
        }

        torch::Tensor tv_loss_forward_cuda(const torch::Tensor& grids) {
            TORCH_CHECK(grids.dim() == 5 && grids.size(1) == 12,
                        "Grids must be [N, 12, L, H, W]");

            const int N = grids.size(0);
            const int L = grids.size(2);
            const int H = grids.size(3);
            const int W = grids.size(4);

            auto tv_loss = torch::zeros({}, grids.options());

            const int threads = 256;
            const int total = N * L * H * W;
            const int blocks = min((total + threads - 1) / threads, 2048);

            tv_loss_forward_kernel<<<blocks, threads>>>(
                grids.data_ptr<float>(),
                tv_loss.data_ptr<float>(),
                N, L, H, W);

            return tv_loss;
        }

        torch::Tensor tv_loss_backward_cuda(
            const torch::Tensor& grids,
            const torch::Tensor& grad_output) {
            const int N = grids.size(0);
            const int L = grids.size(2);
            const int H = grids.size(3);
            const int W = grids.size(4);

            auto grad_grids = torch::zeros_like(grids);

            const size_t total = (size_t)N * 12 * L * H * W;
            const int threads = 256;
            const int blocks = min((int)((total + threads - 1) / threads), 2048);

            tv_loss_backward_kernel<<<blocks, threads>>>(
                grids.data_ptr<float>(),
                grad_output.item<float>(),
                grad_grids.data_ptr<float>(),
                N, L, H, W);

            return grad_grids;
        }

    } // namespace bilateral_grid
} // namespace gs