/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <torch/torch.h>
#include <tuple>

namespace fast_gs::rasterization {

    struct FastGSSettings {
        torch::Tensor cam_position;
        int active_sh_bases;
        int width;
        int height;
        float focal_x;
        float focal_y;
        float center_x;
        float center_y;
        float near_plane;
        float far_plane;
    };

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int, int>
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

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    backward_wrapper(
        torch::Tensor& densification_info,
        const torch::Tensor& grad_image,
        const torch::Tensor& grad_alpha,
        const torch::Tensor& image,
        const torch::Tensor& alpha,
        const torch::Tensor& means,
        const torch::Tensor& scales_raw,
        const torch::Tensor& rotations_raw,
        const torch::Tensor& sh_coefficients_rest,
        const torch::Tensor& per_primitive_buffers,
        const torch::Tensor& per_tile_buffers,
        const torch::Tensor& per_instance_buffers,
        const torch::Tensor& per_bucket_buffers,
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
        const float far_plane,
        const int n_visible_primitives,
        const int n_instances,
        const int n_buckets,
        const int primitive_primitive_indices_selector,
        const int instance_primitive_indices_selector);

} // namespace fast_gs::rasterization
