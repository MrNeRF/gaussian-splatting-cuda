/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gs_rasterizer.hpp"
#include "rasterization_api.h"

namespace gs::rendering {

    struct FastGSSettings {
        torch::Tensor w2c;
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

    static std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& means,                // [N, 3]
        const torch::Tensor& scales_raw,           // [N, 3]
        const torch::Tensor& rotations_raw,        // [N, 4]
        const torch::Tensor& opacities_raw,        // [N, 1]
        const torch::Tensor& sh_coefficients_0,    // [N, 1, 3]
        const torch::Tensor& sh_coefficients_rest, // [C, B-1, 3]
        const FastGSSettings& settings) {          // rasterizer settings

        return forward_wrapper(
            means,
            scales_raw,
            rotations_raw,
            opacities_raw,
            sh_coefficients_0,
            sh_coefficients_rest,
            settings.w2c,
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
    }

    using torch::indexing::None;
    using torch::indexing::Slice;

    torch::Tensor rasterize(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color) {

        // Get camera parameters
        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        FastGSSettings settings{
            .w2c = viewpoint_camera.world_view_transform(),
            .cam_position = viewpoint_camera.cam_position(),
            .active_sh_bases = active_sh_bases,
            .width = viewpoint_camera.image_width(),
            .height = viewpoint_camera.image_height(),
            .focal_x = fx,
            .focal_y = fy,
            .center_x = cx,
            .center_y = cy,
            .near_plane = near_plane,
            .far_plane = far_plane};
        auto [image, alpha] = forward(
            gaussian_model.means(),
            gaussian_model.scaling_raw(),
            gaussian_model.rotation_raw(),
            gaussian_model.opacity_raw(),
            gaussian_model.sh0(),
            gaussian_model.shN(),
            settings);

        // Manually blend the background since the forward pass does not support it
        torch::Tensor bg = bg_color.unsqueeze(1).unsqueeze(2); // [3, 1, 1]
        torch::Tensor blended_image = image + (1.0f - alpha) * bg;

        // Clamp the image to [0, 1] range for consistency with the original rasterize
        blended_image = torch::clamp(blended_image, 0.0f, 1.0f);

        return blended_image;
    }

} // namespace gs::rendering