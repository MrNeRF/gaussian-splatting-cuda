/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backward.h"
#include "forward.h"
#include "helper_math.h"
#include "rasterization_api.h"
#include "rasterization_config.h"
#include "torch_utils.h"
#include <functional>
#include <stdexcept>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int, int>
fast_gs::rasterization::forward_wrapper(
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
    torch::Tensor per_bucket_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_primitive_buffers_func = resize_function_wrapper(per_primitive_buffers);
    const std::function<char*(size_t)> per_tile_buffers_func = resize_function_wrapper(per_tile_buffers);
    const std::function<char*(size_t)> per_instance_buffers_func = resize_function_wrapper(per_instance_buffers);
    const std::function<char*(size_t)> per_bucket_buffers_func = resize_function_wrapper(per_bucket_buffers);

    auto [n_visible_primitives, n_instances, n_buckets, primitive_primitive_indices_selector, instance_primitive_indices_selector] = forward(
        per_primitive_buffers_func,
        per_tile_buffers_func,
        per_instance_buffers_func,
        per_bucket_buffers_func,
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

    return {
        image, alpha,
        per_primitive_buffers, per_tile_buffers, per_instance_buffers, per_bucket_buffers,
        n_visible_primitives, n_instances, n_buckets,
        primitive_primitive_indices_selector, instance_primitive_indices_selector};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fast_gs::rasterization::backward_wrapper(
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
    const int instance_primitive_indices_selector) {
    const int n_primitives = means.size(0);
    const int total_bases_sh_rest = sh_coefficients_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor grad_means = torch::zeros({n_primitives, 3}, float_options);
    torch::Tensor grad_scales_raw = torch::zeros({n_primitives, 3}, float_options);
    torch::Tensor grad_rotations_raw = torch::zeros({n_primitives, 4}, float_options);
    torch::Tensor grad_opacities_raw = torch::zeros({n_primitives, 1}, float_options);
    torch::Tensor grad_sh_coefficients_0 = torch::zeros({n_primitives, 1, 3}, float_options);
    torch::Tensor grad_sh_coefficients_rest = torch::zeros({n_primitives, total_bases_sh_rest, 3}, float_options);
    torch::Tensor grad_mean2d_helper = torch::zeros({n_primitives, 2}, float_options);
    torch::Tensor grad_conic_helper = torch::zeros({n_primitives, 3}, float_options);
    torch::Tensor grad_w2c = torch::Tensor();
    if (w2c.requires_grad()) {
        grad_w2c = torch::zeros_like(w2c, float_options);
    }

    const bool update_densification_info = densification_info.size(0) > 0;

    backward(
        grad_image.data_ptr<float>(),
        grad_alpha.data_ptr<float>(),
        image.data_ptr<float>(),
        alpha.data_ptr<float>(),
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales_raw.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations_raw.data_ptr<float>()),
        reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()),
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<char*>(per_primitive_buffers.data_ptr()),
        reinterpret_cast<char*>(per_tile_buffers.data_ptr()),
        reinterpret_cast<char*>(per_instance_buffers.data_ptr()),
        reinterpret_cast<char*>(per_bucket_buffers.data_ptr()),
        reinterpret_cast<float3*>(grad_means.data_ptr<float>()),
        reinterpret_cast<float3*>(grad_scales_raw.data_ptr<float>()),
        reinterpret_cast<float4*>(grad_rotations_raw.data_ptr<float>()),
        reinterpret_cast<float*>(grad_opacities_raw.data_ptr<float>()),
        reinterpret_cast<float3*>(grad_sh_coefficients_0.data_ptr<float>()),
        reinterpret_cast<float3*>(grad_sh_coefficients_rest.data_ptr<float>()),
        reinterpret_cast<float2*>(grad_mean2d_helper.data_ptr<float>()),
        grad_conic_helper.data_ptr<float>(),
        w2c.requires_grad() ? reinterpret_cast<float4*>(grad_w2c.data_ptr<float>()) : nullptr,
        update_densification_info ? densification_info.data_ptr<float>() : nullptr,
        n_primitives,
        n_visible_primitives,
        n_instances,
        n_buckets,
        primitive_primitive_indices_selector,
        instance_primitive_indices_selector,
        active_sh_bases,
        total_bases_sh_rest,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y);

    return {grad_means, grad_scales_raw, grad_rotations_raw, grad_opacities_raw, grad_sh_coefficients_0, grad_sh_coefficients_rest, grad_w2c};
}
