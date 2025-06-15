#include "kernels/bilateral_grid.cuh"
#include <cuda_runtime.h>

namespace gs {
    namespace bilateral_grid {

        // RGB to grayscale conversion constants
        __constant__ float kC2G_bwd[3] = {0.299f, 0.587f, 0.114f};

        __global__ void slice_backward_kernel(
            const float* __restrict__ grid,        // [12, L, H, W]
            const float* __restrict__ rgb,         // [h, w, 3]
            const float* __restrict__ grad_output, // [h, w, 3]
            float* __restrict__ grad_grid,         // [12, L, H, W]
            float* __restrict__ grad_rgb,          // [h, w, 3]
            int L, int H, int W,
            int h, int w) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = gridDim.x * blockDim.x;
            int total = h * w;

            for (int pixel_idx = idx; pixel_idx < total; pixel_idx += stride) {
                int hi = pixel_idx / w;
                int wi = pixel_idx % w;

                // Load data
                int rgb_offset = pixel_idx * 3;
                float sr = rgb[rgb_offset + 0];
                float sg = rgb[rgb_offset + 1];
                float sb = rgb[rgb_offset + 2];

                // Grid coordinates (uniform sampling)
                float gx = (float)wi / (float)(w - 1);
                float gy = (float)hi / (float)(h - 1);
                float gz = kC2G_bwd[0] * sr + kC2G_bwd[1] * sg + kC2G_bwd[2] * sb;

                float x = gx * (W - 1);
                float y = gy * (H - 1);
                float z = gz * (L - 1);

                // Compute bounds
                int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
                int x1 = min(x0 + 1, W - 1);
                int y1 = min(y0 + 1, H - 1);
                int z1 = min(max(z0 + 1, 0), L - 1);
                z0 = max(z0, 0);

                // Fractional parts
                float fx = x - x0, fy = y - y0, fz = z - z0;

                // Precompute weights
                float w[8];
                w[0] = (1 - fx) * (1 - fy) * (1 - fz);
                w[1] = fx * (1 - fy) * (1 - fz);
                w[2] = (1 - fx) * fy * (1 - fz);
                w[3] = fx * fy * (1 - fz);
                w[4] = (1 - fx) * (1 - fy) * fz;
                w[5] = fx * (1 - fy) * fz;
                w[6] = (1 - fx) * fy * fz;
                w[7] = fx * fy * fz;

                // Corner positions
                int cx[8] = {x0, x1, x0, x1, x0, x1, x0, x1};
                int cy[8] = {y0, y0, y1, y1, y0, y0, y1, y1};
                int cz[8] = {z0, z0, z0, z0, z1, z1, z1, z1};

                // Read upstream gradient
                float dr = grad_output[rgb_offset + 0];
                float dg = grad_output[rgb_offset + 1];
                float db = grad_output[rgb_offset + 2];

                // Gradient w.r.t RGB
                float vr = 0.0f, vg = 0.0f, vb = 0.0f;

// Process channels
#pragma unroll 3
                for (int di = 0; di < 3; di++) {
                    float gout = (di == 0 ? dr : di == 1 ? dg
                                                         : db);

#pragma unroll 4
                    for (int si = 0; si < 4; si++) {
                        int ci = di * 4 + si;
                        float r_coeff = (si == 0 ? sr : si == 1 ? sg
                                                    : si == 2   ? sb
                                                                : 1.0f);
                        float grad_weight = r_coeff * gout;

// Update grid gradients
#pragma unroll 8
                        for (int corner = 0; corner < 8; corner++) {
                            int grid_idx = ci * L * H * W +
                                           (cz[corner] * H + cy[corner]) * W + cx[corner];
                            atomicAdd(grad_grid + grid_idx, w[corner] * grad_weight);

                            // Accumulate for RGB gradient
                            if (si < 3) {
                                float val = grid[grid_idx];
                                if (si == 0)
                                    vr += val * w[corner] * gout;
                                else if (si == 1)
                                    vg += val * w[corner] * gout;
                                else
                                    vb += val * w[corner] * gout;
                            }
                        }
                    }
                }

                // Gradient w.r.t. grayscale (through z coordinate)
                float gz_grad = 0.0f;

                // Spatial derivatives
                float dwdz[8] = {
                    -(1 - fx) * (1 - fy), -fx * (1 - fy),
                    -(1 - fx) * fy, -fx * fy,
                    (1 - fx) * (1 - fy), fx * (1 - fy),
                    (1 - fx) * fy, fx * fy};

#pragma unroll 8
                for (int corner = 0; corner < 8; corner++) {
                    float trilerp = 0.0f;

#pragma unroll 12
                    for (int ci = 0; ci < 12; ci++) {
                        int grid_idx = ci * L * H * W +
                                       (cz[corner] * H + cy[corner]) * W + cx[corner];
                        float v = grid[grid_idx];

                        int si = ci % 4, di = ci / 4;
                        float r_coeff = (si == 0 ? sr : si == 1 ? sg
                                                    : si == 2   ? sb
                                                                : 1.0f);
                        float gout = (di == 0 ? dr : di == 1 ? dg
                                                             : db);
                        trilerp += v * r_coeff * gout;
                    }
                    gz_grad += dwdz[corner] * (L - 1) * trilerp;
                }

                // Apply discontinuity masking
                gz_grad *= (float)(z0 != z && z1 != z);

                // Save gradients
                grad_rgb[rgb_offset + 0] = vr + kC2G_bwd[0] * gz_grad;
                grad_rgb[rgb_offset + 1] = vg + kC2G_bwd[1] * gz_grad;
                grad_rgb[rgb_offset + 2] = vb + kC2G_bwd[2] * gz_grad;
            }
        }

        std::tuple<torch::Tensor, torch::Tensor> slice_backward_cuda(
            const torch::Tensor& grid,
            const torch::Tensor& rgb,
            const torch::Tensor& grad_output) {
            const int h = rgb.size(0);
            const int w = rgb.size(1);
            const int L = grid.size(1);
            const int H = grid.size(2);
            const int W = grid.size(3);

            // Initialize gradient tensors
            auto grad_grid = torch::zeros_like(grid);
            auto grad_rgb = torch::empty_like(rgb);

            // Launch kernel
            const int threads = 256;
            const int total = h * w;
            const int blocks = min((total + threads - 1) / threads, 65535);

            slice_backward_kernel<<<blocks, threads>>>(
                grid.data_ptr<float>(),
                rgb.data_ptr<float>(),
                grad_output.data_ptr<float>(),
                grad_grid.data_ptr<float>(),
                grad_rgb.data_ptr<float>(),
                L, H, W, h, w);

            return std::make_tuple(grad_grid, grad_rgb);
        }

    } // namespace bilateral_grid
} // namespace gs