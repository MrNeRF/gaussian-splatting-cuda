/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fast_rasterizer.hpp"
#include "fast_rasterizer_autograd.hpp"

namespace gs::training {
    using torch::indexing::None;
    using torch::indexing::Slice;

    RenderOutput fast_rasterize(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color) {
        // Get camera parameters
        const int width = static_cast<int>(viewpoint_camera.image_width());
        const int height = static_cast<int>(viewpoint_camera.image_height());
        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        // Get Gaussian parameters
        auto means = gaussian_model.means();
        auto raw_opacities = gaussian_model.opacity_raw();
        auto raw_scales = gaussian_model.scaling_raw();
        auto raw_rotations = gaussian_model.rotation_raw();
        auto sh0 = gaussian_model.sh0();
        auto shN = gaussian_model.shN();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        fast_gs::rasterization::FastGSSettings settings;
        auto w2c = viewpoint_camera.world_view_transform();
        settings.cam_position = viewpoint_camera.cam_position();
        settings.active_sh_bases = active_sh_bases;
        settings.width = width;
        settings.height = height;
        settings.focal_x = fx;
        settings.focal_y = fy;
        settings.center_x = cx;
        settings.center_y = cy;
        settings.near_plane = near_plane;
        settings.far_plane = far_plane;

        auto raster_outputs = FastGSRasterize::apply(
            means,
            raw_scales,
            raw_rotations,
            raw_opacities,
            sh0,
            shN,
            w2c,
            gaussian_model._densification_info,
            settings);

        RenderOutput output;
        output.image = raster_outputs[0];
        output.alpha = raster_outputs[1];

        output.image = output.image + (1.0f - output.alpha) * bg_color.unsqueeze(-1).unsqueeze(-1);

        return output;
    }
} // namespace gs::training
