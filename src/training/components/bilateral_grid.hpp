/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <torch/torch.h>

namespace gs::training {

    class BilateralGrid {
    public:
        BilateralGrid(int num_images, int grid_W = 16, int grid_H = 16, int grid_L = 8);

        // Apply bilateral grid to rendered image
        torch::Tensor apply(const torch::Tensor& rgb, int image_idx);

        // Compute total variation loss
        torch::Tensor tv_loss() const;

        // Get parameters for optimizer
        torch::Tensor parameters() { return grids_; }
        const torch::Tensor& parameters() const { return grids_; }

        // Grid dimensions
        int grid_width() const { return grid_width_; }
        int grid_height() const { return grid_height_; }
        int grid_guidance() const { return grid_guidance_; }

    private:
        torch::Tensor grids_; // [N, 12, L, H, W]
        int num_images_;
        int grid_width_;
        int grid_height_;
        int grid_guidance_;
    };

} // namespace gs::training