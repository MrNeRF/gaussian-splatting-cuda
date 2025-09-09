/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <torch/torch.h>
#include <tuple>

namespace gs::rendering {

    std::tuple<torch::Tensor, torch::Tensor>
    forward_wrapper(
        const torch::Tensor& means,
        const torch::Tensor& scales_raw,
        const torch::Tensor& rotations_raw,
        const torch::Tensor& opacities_raw,
        const torch::Tensor& sh_coefficients_0,
        const torch::Tensor& sh_coefficients_rest,
        const torch::Tensor& w2c,
        const torch::Tensor& cam_position,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane);
} // namespace gs::rendering
