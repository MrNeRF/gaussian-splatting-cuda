/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fast_rasterizer_autograd.hpp"

namespace gs::training {
    // FastGSRasterize implementation
    torch::autograd::tensor_list FastGSRasterize::forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& means,                // [N, 3]
        const torch::Tensor& scales_raw,           // [N, 3]
        const torch::Tensor& rotations_raw,        // [N, 4]
        const torch::Tensor& opacities_raw,        // [N, 1]
        const torch::Tensor& sh_coefficients_0,    // [N, 1, 3]
        const torch::Tensor& sh_coefficients_rest, // [C, B-1, 3]
        const torch::Tensor& w2c,                  // [C, 4, 4]
        torch::Tensor& densification_info,         // [2, N] or empty tensor
        const fast_gs::rasterization::FastGSSettings& settings) {
        // rasterizer settings

        auto outputs = fast_gs::rasterization::forward_wrapper(
            means,
            scales_raw,
            rotations_raw,
            opacities_raw,
            sh_coefficients_0,
            sh_coefficients_rest,
            w2c,
            settings.cam_position,
            settings.active_sh_bases,
            settings.width,
            settings.height,
            settings.focal_x,
            settings.focal_y,
            settings.center_x,
            settings.center_y,
            settings.near_plane,
            settings.far_plane);

        auto image = std::get<0>(outputs);
        auto alpha = std::get<1>(outputs);
        auto per_primitive_buffers = std::get<2>(outputs);
        auto per_tile_buffers = std::get<3>(outputs);
        auto per_instance_buffers = std::get<4>(outputs);
        auto per_bucket_buffers = std::get<5>(outputs);
        int n_visible_primitives = std::get<6>(outputs);
        int n_instances = std::get<7>(outputs);
        int n_buckets = std::get<8>(outputs);
        int primitive_primitive_indices_selector = std::get<9>(outputs);
        int instance_primitive_indices_selector = std::get<10>(outputs);

        // Mark non-differentiable tensors
        ctx->mark_non_differentiable({per_primitive_buffers,
                                      per_tile_buffers,
                                      per_instance_buffers,
                                      per_bucket_buffers,
                                      densification_info});

        // Save for backward
        ctx->save_for_backward({image,
                                alpha,
                                means,
                                scales_raw,
                                rotations_raw,
                                sh_coefficients_rest,
                                per_primitive_buffers,
                                per_tile_buffers,
                                per_instance_buffers,
                                per_bucket_buffers,
                                w2c,
                                densification_info});

        ctx->saved_data["cam_position"] = settings.cam_position;
        ctx->saved_data["active_sh_bases"] = settings.active_sh_bases;
        ctx->saved_data["width"] = settings.width;
        ctx->saved_data["height"] = settings.height;
        ctx->saved_data["focal_x"] = settings.focal_x;
        ctx->saved_data["focal_y"] = settings.focal_y;
        ctx->saved_data["center_x"] = settings.center_x;
        ctx->saved_data["center_y"] = settings.center_y;
        ctx->saved_data["near_plane"] = settings.near_plane;
        ctx->saved_data["far_plane"] = settings.far_plane;
        ctx->saved_data["n_visible_primitives"] = n_visible_primitives;
        ctx->saved_data["n_instances"] = n_instances;
        ctx->saved_data["n_buckets"] = n_buckets;
        ctx->saved_data["primitive_primitive_indices_selector"] = primitive_primitive_indices_selector;
        ctx->saved_data["instance_primitive_indices_selector"] = instance_primitive_indices_selector;

        return {image, alpha};
    }

    torch::autograd::tensor_list FastGSRasterize::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
        auto grad_image = grad_outputs[0];
        auto grad_alpha = grad_outputs[1];

        auto saved = ctx->get_saved_variables();
        const torch::Tensor& image = saved[0];
        const torch::Tensor& alpha = saved[1];
        const torch::Tensor& means = saved[2];
        const torch::Tensor& scales_raw = saved[3];
        const torch::Tensor& rotations_raw = saved[4];
        const torch::Tensor& sh_coefficients_rest = saved[5];
        const torch::Tensor& per_primitive_buffers = saved[6];
        const torch::Tensor& per_tile_buffers = saved[7];
        const torch::Tensor& per_instance_buffers = saved[8];
        const torch::Tensor& per_bucket_buffers = saved[9];
        const torch::Tensor& w2c = saved[10];
        torch::Tensor& densification_info = saved[11];

        auto outputs = fast_gs::rasterization::backward_wrapper(
            densification_info,
            grad_image,
            grad_alpha,
            image,
            alpha,
            means,
            scales_raw,
            rotations_raw,
            sh_coefficients_rest,
            per_primitive_buffers,
            per_tile_buffers,
            per_instance_buffers,
            per_bucket_buffers,
            w2c,
            ctx->saved_data["cam_position"].toTensor(),
            ctx->saved_data["active_sh_bases"].toInt(),
            ctx->saved_data["width"].toInt(),
            ctx->saved_data["height"].toInt(),
            static_cast<float>(ctx->saved_data["focal_x"].toDouble()),
            static_cast<float>(ctx->saved_data["focal_y"].toDouble()),
            static_cast<float>(ctx->saved_data["center_x"].toDouble()),
            static_cast<float>(ctx->saved_data["center_y"].toDouble()),
            static_cast<float>(ctx->saved_data["near_plane"].toDouble()),
            static_cast<float>(ctx->saved_data["far_plane"].toDouble()),
            ctx->saved_data["n_visible_primitives"].toInt(),
            ctx->saved_data["n_instances"].toInt(),
            ctx->saved_data["n_buckets"].toInt(),
            ctx->saved_data["primitive_primitive_indices_selector"].toInt(),
            ctx->saved_data["instance_primitive_indices_selector"].toInt());

        auto grad_means = std::get<0>(outputs);
        auto grad_scales_raw = std::get<1>(outputs);
        auto grad_rotations_raw = std::get<2>(outputs);
        auto grad_opacities_raw = std::get<3>(outputs);
        auto grad_sh_coefficients_0 = std::get<4>(outputs);
        auto grad_sh_coefficients_rest = std::get<5>(outputs);
        auto grad_w2c = std::get<6>(outputs);

        return {
            grad_means,
            grad_scales_raw,
            grad_rotations_raw,
            grad_opacities_raw,
            grad_sh_coefficients_0,
            grad_sh_coefficients_rest,
            grad_w2c,
            torch::Tensor(), // densification_info (no gradient)
            torch::Tensor(), // settings (no gradient)
        };
    }
} // namespace gs::training
