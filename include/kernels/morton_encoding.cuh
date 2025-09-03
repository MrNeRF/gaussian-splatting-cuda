/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */


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
