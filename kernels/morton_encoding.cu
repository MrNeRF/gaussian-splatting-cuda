// Morton encoding implementation based on:
// 1. https://github.com/m-schuetz/compute_rasterizer/blob/f2cbb658e6bf58407c385c75d21f3f615f11d5c9/tools/sort_points/Sort_Frugal/src/main.cpp#L79
// 2. https://gitlab.inria.fr/sibr/sibr_core/-/blob/gaussian_code_release_linux/src/projects/gaussianviewer/renderer/GaussianView.cpp?ref_type=heads#L90
// 3. https://github.com/nerficg-project/cuda-utils/tree/main/MortonEncoding

#include "kernels/morton_encoding.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace gs {

    __device__ __forceinline__ uint64_t splitBy3(uint32_t a) {
        uint64_t x = a & 0x1fffff;
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8) & 0x100f00f00f00f00f;
        x = (x | x << 4) & 0x10c30c30c30c30c3;
        x = (x | x << 2) & 0x1249249249249249;
        return x;
    }

    __global__ void morton_encode_cu(
        const float3* __restrict__ positions,
        const float3* __restrict__ minimum_coordinates,
        const float* __restrict__ cube_size,
        int64_t* __restrict__ morton_encoding,
        const int n_positions) {

        const int position_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (position_idx >= n_positions)
            return;

        const float3 position = positions[position_idx];
        const float3 minimum_coordinate = minimum_coordinates[0];

        // Could use float instead of double if performance is critical
        const double size = double(cube_size[0]);

        const double normalized_x = double(position.x - minimum_coordinate.x) / size;
        const double normalized_y = double(position.y - minimum_coordinate.y) / size;
        const double normalized_z = double(position.z - minimum_coordinate.z) / size;

        constexpr double factor = 2097151.0; // 2^21 - 1

        const uint32_t x = static_cast<uint32_t>(normalized_x * factor);
        const uint32_t y = static_cast<uint32_t>(normalized_y * factor);
        const uint32_t z = static_cast<uint32_t>(normalized_z * factor);

        const uint64_t morton_code = splitBy3(x) | (splitBy3(y) << 1) | (splitBy3(z) << 2);

        // Convert to signed int64 for PyTorch compatibility
        constexpr int64_t int64_min = std::numeric_limits<int64_t>::min();
        const int64_t morton_code_torch = static_cast<int64_t>(morton_code) + int64_min;

        morton_encoding[position_idx] = morton_code_torch;
    }

    torch::Tensor morton_encode(const torch::Tensor& positions) {
        TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
                    "Positions must have shape [N, 3]");
        TORCH_CHECK(positions.is_cuda(), "Positions must be on CUDA");
        TORCH_CHECK(positions.dtype() == torch::kFloat32, "Positions must be float32");

        const int n_positions = positions.size(0);

        // Compute bounding box
        auto min_vals = std::get<0>(positions.min(0));
        auto max_vals = std::get<0>(positions.max(0));
        auto range = max_vals - min_vals;
        auto cube_size = std::get<0>(range.max(0));

        // Add small epsilon to avoid division by zero
        cube_size = torch::clamp_min(cube_size, 1e-7f);

        // Allocate output
        auto morton_encoding = torch::empty({n_positions},
                                            positions.options().dtype(torch::kInt64));

        // Launch kernel
        constexpr int block_size = 256;
        const int grid_size = (n_positions + block_size - 1) / block_size;

        morton_encode_cu<<<grid_size, block_size>>>(
            reinterpret_cast<const float3*>(positions.contiguous().data_ptr<float>()),
            reinterpret_cast<const float3*>(min_vals.contiguous().data_ptr<float>()),
            cube_size.data_ptr<float>(),
            morton_encoding.data_ptr<int64_t>(),
            n_positions);

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error in morton_encode: ") +
                                     cudaGetErrorString(err));
        }

        return morton_encoding;
    }

    torch::Tensor morton_sort_indices(const torch::Tensor& morton_codes) {
        TORCH_CHECK(morton_codes.dim() == 1, "Morton codes must be 1D tensor");
        TORCH_CHECK(morton_codes.is_cuda(), "Morton codes must be on CUDA");

        // Use torch::argsort to get the indices that would sort the morton codes
        return torch::argsort(morton_codes, 0, false); // ascending order
    }

} // namespace gs
