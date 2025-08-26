#include "kernels/bilateral_grid.cuh"
#include <cuda_runtime.h>

namespace gs {
    namespace bilateral_grid {

        // RGB to grayscale conversion constants
        __constant__ float kC2G[3] = {0.299f, 0.587f, 0.114f};

        __global__ void slice_forward_kernel(
            const float* __restrict__ grid, // [12, L, H, W]
            const float* __restrict__ rgb,  // [h, w, 3]
            float* __restrict__ output,     // [h, w, 3]
            int L, int H, int W,
            int h, int w) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= h * w)
                return;

            int hi = idx / w;
            int wi = idx % w;

            // Load RGB values
            float3 color;
            int rgb_idx = idx * 3;
            color.x = rgb[rgb_idx + 0];
            color.y = rgb[rgb_idx + 1];
            color.z = rgb[rgb_idx + 2];

            // Compute grid coordinates (uniform sampling)
            float gx = (float)wi / (float)(w - 1);
            float gy = (float)hi / (float)(h - 1);
            float gz = kC2G[0] * color.x + kC2G[1] * color.y + kC2G[2] * color.z;

            float x = gx * (W - 1);
            float y = gy * (H - 1);
            float z = gz * (L - 1);

            // Trilinear interpolation setup
            int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
            int x1 = min(x0 + 1, W - 1);
            int y1 = min(y0 + 1, H - 1);
            int z1 = min(max(z0 + 1, 0), L - 1);
            z0 = max(z0, 0);

            float fx = x - x0, fy = y - y0, fz = z - z0;

            // Apply affine transformation
            float3 result = make_float3(0.0f, 0.0f, 0.0f);

#pragma unroll
            for (int ci = 0; ci < 12; ci++) {
                // Trilinear interpolation
                int base = ci * L * H * W;
                float val = 0.0f;

                // 8 corner values
                val += grid[base + (z0 * H + y0) * W + x0] * (1 - fx) * (1 - fy) * (1 - fz);
                val += grid[base + (z0 * H + y0) * W + x1] * fx * (1 - fy) * (1 - fz);
                val += grid[base + (z0 * H + y1) * W + x0] * (1 - fx) * fy * (1 - fz);
                val += grid[base + (z0 * H + y1) * W + x1] * fx * fy * (1 - fz);
                val += grid[base + (z1 * H + y0) * W + x0] * (1 - fx) * (1 - fy) * fz;
                val += grid[base + (z1 * H + y0) * W + x1] * fx * (1 - fy) * fz;
                val += grid[base + (z1 * H + y1) * W + x0] * (1 - fx) * fy * fz;
                val += grid[base + (z1 * H + y1) * W + x1] * fx * fy * fz;

                // Apply to appropriate channel
                int si = ci % 4; // source index
                int di = ci / 4; // destination index

                float coeff = (si == 0) ? color.x : (si == 1) ? color.y
                                                : (si == 2)   ? color.z
                                                              : 1.0f;

                if (di == 0)
                    result.x += val * coeff;
                else if (di == 1)
                    result.y += val * coeff;
                else
                    result.z += val * coeff;
            }

            // Write output
            output[rgb_idx + 0] = result.x;
            output[rgb_idx + 1] = result.y;
            output[rgb_idx + 2] = result.z;
        }

        void slice_forward_cuda(
            const torch::Tensor& grid,
            const torch::Tensor& rgb,
            torch::Tensor& output,
            bool use_uniform_coords) {
            const int h = rgb.size(0);
            const int w = rgb.size(1);
            const int L = grid.size(1);
            const int H = grid.size(2);
            const int W = grid.size(3);

            const int threads = 256;
            const int blocks = (h * w + threads - 1) / threads;

            slice_forward_kernel<<<blocks, threads>>>(
                grid.data_ptr<float>(),
                rgb.data_ptr<float>(),
                output.data_ptr<float>(),
                L, H, W, h, w);
        }

    } // namespace bilateral_grid
} // namespace gs