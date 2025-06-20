// kernels/cuda_gl_interop.cu
#include "config.h"

#ifdef CUDA_GL_INTEROP_ENABLED

#include <cuda_runtime.h>
#include <torch/torch.h>

namespace gs {

    // Kernel for converting RGB float to RGBA uint8
    __global__ void convertRGBFloatToRGBAUint8(
        const float* __restrict__ rgb,
        uint8_t* __restrict__ rgba,
        int width, int height) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            int idx = y * width + x;
            int rgb_idx = idx * 3;
            int rgba_idx = idx * 4;

            // Convert float [0,1] to uint8 [0,255] with clamping
            rgba[rgba_idx + 0] = min(255, max(0, __float2int_rn(rgb[rgb_idx + 0] * 255.0f)));
            rgba[rgba_idx + 1] = min(255, max(0, __float2int_rn(rgb[rgb_idx + 1] * 255.0f)));
            rgba[rgba_idx + 2] = min(255, max(0, __float2int_rn(rgb[rgb_idx + 2] * 255.0f)));
            rgba[rgba_idx + 3] = 255; // Alpha
        }
    }

    // Kernel for converting RGBA float to RGBA uint8
    __global__ void convertRGBAFloatToRGBAUint8(
        const float* __restrict__ rgba_in,
        uint8_t* __restrict__ rgba_out,
        int width, int height) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            int idx = (y * width + x) * 4;

            // Convert float [0,1] to uint8 [0,255] with clamping
            rgba_out[idx + 0] = min(255, max(0, __float2int_rn(rgba_in[idx + 0] * 255.0f)));
            rgba_out[idx + 1] = min(255, max(0, __float2int_rn(rgba_in[idx + 1] * 255.0f)));
            rgba_out[idx + 2] = min(255, max(0, __float2int_rn(rgba_in[idx + 2] * 255.0f)));
            rgba_out[idx + 3] = min(255, max(0, __float2int_rn(rgba_in[idx + 3] * 255.0f)));
        }
    }

    // Kernel for flipping image vertically (OpenGL uses bottom-left origin)
    template <typename T>
    __global__ void flipVertical(
        const T* __restrict__ input,
        T* __restrict__ output,
        int width, int height, int channels) {

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            int flipped_y = height - 1 - y;
            int src_idx = (y * width + x) * channels;
            int dst_idx = (flipped_y * width + x) * channels;

#pragma unroll
            for (int c = 0; c < channels; ++c) {
                output[dst_idx + c] = input[src_idx + c];
            }
        }
    }

    // Helper function to convert tensor format for OpenGL
    torch::Tensor convertToRGBA8(const torch::Tensor& input, bool flip_vertical) {
        TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
        TORCH_CHECK(input.dim() == 3, "Input must be [H, W, C]");

        const int h = input.size(0);
        const int w = input.size(1);
        const int c = input.size(2);

        // Already in correct format
        if (c == 4 && input.dtype() == torch::kUInt8 && !flip_vertical) {
            return input;
        }

        // Allocate output
        auto output = torch::empty({h, w, 4},
                                   input.options().dtype(torch::kUInt8));

        // Configure kernel launch
        dim3 block(16, 16);
        dim3 grid((w + block.x - 1) / block.x,
                  (h + block.y - 1) / block.y);

        // Conversion based on input format
        if (c == 3 && input.dtype() == torch::kFloat32) {
            if (flip_vertical) {
                // Convert and flip in one pass
                auto temp = torch::empty_like(output);
                convertRGBFloatToRGBAUint8<<<grid, block>>>(
                    input.data_ptr<float>(),
                    temp.data_ptr<uint8_t>(),
                    w, h);
                flipVertical<uint8_t><<<grid, block>>>(
                    temp.data_ptr<uint8_t>(),
                    output.data_ptr<uint8_t>(),
                    w, h, 4);
            } else {
                convertRGBFloatToRGBAUint8<<<grid, block>>>(
                    input.data_ptr<float>(),
                    output.data_ptr<uint8_t>(),
                    w, h);
            }
        } else if (c == 4 && input.dtype() == torch::kFloat32) {
            if (flip_vertical) {
                auto temp = torch::empty_like(output);
                convertRGBAFloatToRGBAUint8<<<grid, block>>>(
                    input.data_ptr<float>(),
                    temp.data_ptr<uint8_t>(),
                    w, h);
                flipVertical<uint8_t><<<grid, block>>>(
                    temp.data_ptr<uint8_t>(),
                    output.data_ptr<uint8_t>(),
                    w, h, 4);
            } else {
                convertRGBAFloatToRGBAUint8<<<grid, block>>>(
                    input.data_ptr<float>(),
                    output.data_ptr<uint8_t>(),
                    w, h);
            }
        } else if (flip_vertical && c == 4 && input.dtype() == torch::kUInt8) {
            flipVertical<uint8_t><<<grid, block>>>(
                input.data_ptr<uint8_t>(),
                output.data_ptr<uint8_t>(),
                w, h, 4);
        } else {
            // Fallback to torch operations
            torch::Tensor rgba_float;
            if (c == 3) {
                rgba_float = torch::cat({input.to(torch::kFloat32),
                                         torch::ones({h, w, 1}, input.options().dtype(torch::kFloat32))},
                                        2);
            } else {
                rgba_float = input.to(torch::kFloat32);
            }

            output = (rgba_float.clamp(0.0f, 1.0f) * 255.0f).to(torch::kUInt8);

            if (flip_vertical) {
                output = output.flip({0});
            }
        }

        // Ensure kernel completion
        cudaDeviceSynchronize();

        return output;
    }

} // namespace gs

#endif // CUDA_GL_INTEROP_ENABLED