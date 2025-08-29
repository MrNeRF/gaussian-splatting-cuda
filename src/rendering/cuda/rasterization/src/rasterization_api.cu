/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "forward.h"
#include "rasterization_api.h"
#include "rasterization_config.h"
#include "torch_utils.h"
#include <functional>
#include <stdexcept>
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
        const float far_plane) {
        // all optimizable tensors must be contiguous CUDA float tensors
        CHECK_INPUT(config::debug, means, "means");
        CHECK_INPUT(config::debug, scales_raw, "scales_raw");
        CHECK_INPUT(config::debug, rotations_raw, "rotations_raw");
        CHECK_INPUT(config::debug, opacities_raw, "opacities_raw");
        CHECK_INPUT(config::debug, sh_coefficients_0, "sh_coefficients_0");
        CHECK_INPUT(config::debug, sh_coefficients_rest, "sh_coefficients_rest");

        const int n_primitives = means.size(0);
        const int total_bases_sh_rest = sh_coefficients_rest.size(1);
        const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
        const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
        torch::Tensor image = torch::empty({3, height, width}, float_options);
        torch::Tensor alpha = torch::empty({1, height, width}, float_options);
        torch::Tensor per_primitive_buffers = torch::empty({0}, byte_options);
        torch::Tensor per_tile_buffers = torch::empty({0}, byte_options);
        torch::Tensor per_instance_buffers = torch::empty({0}, byte_options);
        torch::Tensor per_bucket_buffers = torch::empty({0}, byte_options); // Still needed internally, but much smaller
        const std::function<char*(size_t)> per_primitive_buffers_func = resize_function_wrapper(per_primitive_buffers);
        const std::function<char*(size_t)> per_tile_buffers_func = resize_function_wrapper(per_tile_buffers);
        const std::function<char*(size_t)> per_instance_buffers_func = resize_function_wrapper(per_instance_buffers);

        forward(
            per_primitive_buffers_func,
            per_tile_buffers_func,
            per_instance_buffers_func,
            reinterpret_cast<float3*>(means.data_ptr<float>()),
            reinterpret_cast<float3*>(scales_raw.data_ptr<float>()),
            reinterpret_cast<float4*>(rotations_raw.data_ptr<float>()),
            opacities_raw.data_ptr<float>(),
            reinterpret_cast<float3*>(sh_coefficients_0.data_ptr<float>()),
            reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()),
            reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
            reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
            image.data_ptr<float>(),
            alpha.data_ptr<float>(),
            n_primitives,
            active_sh_bases,
            total_bases_sh_rest,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            near_plane,
            far_plane);

        return {image, alpha};
    }

} // namespace gs::rendering