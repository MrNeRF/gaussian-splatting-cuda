// Morton encoding implementation based on:
// 1. https://github.com/m-schuetz/compute_rasterizer/blob/f2cbb658e6bf58407c385c75d21f3f615f11d5c9/tools/sort_points/Sort_Frugal/src/main.cpp#L79
// 2. https://gitlab.inria.fr/sibr/sibr_core/-/blob/gaussian_code_release_linux/src/projects/gaussianviewer/renderer/GaussianView.cpp?ref_type=heads#L90
// 3. https://github.com/nerficg-project/cuda-utils/tree/main/MortonEncoding

#pragma once

#include <torch/torch.h>

namespace gs {
    /**
     * @brief Compute Morton codes for 3D positions
     *
     * This function encodes 3D positions into Morton codes (Z-order curve) for
     * spatial sorting. This improves cache locality during rendering.
     *
     * @param positions Tensor of shape [N, 3] containing 3D positions
     * @return Tensor of shape [N] containing Morton codes as int64
     */
    torch::Tensor morton_encode(const torch::Tensor& positions);

    /**
     * @brief Sort indices by Morton codes
     *
     * @param morton_codes Tensor of Morton codes
     * @return Tensor of indices that would sort the Morton codes
     */
    torch::Tensor morton_sort_indices(const torch::Tensor& morton_codes);
} // namespace gs
