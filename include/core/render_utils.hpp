// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/rasterizer.hpp"
#include "core/splat_data.hpp"
#include "core/gsplat_rasterizer.hpp"
#include <cmath>
#include <torch/torch.h>

struct RenderOutput {
    torch::Tensor image;         // rendered_image
    torch::Tensor viewspace_pts; // means2D
    torch::Tensor visibility;    // radii > 0
    torch::Tensor radii;         // per-Gaussian projected radius
};

inline RenderOutput render(Camera& viewpoint_camera,
                           const SplatData& gaussian_model,
                           torch::Tensor& bg_color,
                           float scaling_modifier = 1.0) {
    // Ensure background tensor (bg_color) is on GPU!
    bg_color = bg_color.to(torch::kCUDA);

    // Set up rasterization configuration
    GaussianRasterizationSettings raster_settings = {
        .image_height = static_cast<int>(viewpoint_camera.Get_image_height()),
        .image_width = static_cast<int>(viewpoint_camera.Get_image_width()),
        .tanfovx = std::tan(viewpoint_camera.Get_FoVx() * 0.5f),
        .tanfovy = std::tan(viewpoint_camera.Get_FoVy() * 0.5f),
        .bg = bg_color,
        .scale_modifier = scaling_modifier,
        .viewmatrix = viewpoint_camera.Get_world_view_transform(),
        .projmatrix = viewpoint_camera.Get_full_proj_transform(),
        .sh_degree = gaussian_model.get_active_sh_degree(),
        .camera_center = viewpoint_camera.Get_camera_center(),
        .prefiltered = false};

    GaussianRasterizer rasterizer = GaussianRasterizer(raster_settings);

    auto means3D = gaussian_model.get_xyz();
    auto means2D = torch::zeros_like(gaussian_model.get_xyz()).requires_grad_(true);
    means2D.retain_grad();
    auto opacity = gaussian_model.get_opacity();

    auto scales = torch::Tensor();
    auto rotations = torch::Tensor();
    auto cov3D_precomp = torch::Tensor();

    scales = gaussian_model.get_scaling();
    rotations = gaussian_model.get_rotation();

    auto shs = torch::Tensor();
    torch::Tensor colors_precomp = torch::Tensor();
    shs = gaussian_model.get_features();

    torch::cuda::synchronize();

    // Rasterize visible Gaussians to image, obtain their radii (on screen).
    auto [rendererd_image, radii] = rasterizer.forward(
        means3D,
        means2D,
        opacity,
        shs,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp);

    // render, viewspace_points, visibility_filter, radii
    return {rendererd_image, means2D, radii > 0, radii};
}

inline RenderOutput render_with_gsplat(Camera& viewpoint_camera,
                                       const SplatData& gaussian_model,
                                       torch::Tensor& bg_color,
                                       float scaling_modifier = 1.0) {
    // Use gsplat backend with proper namespace
    auto gsplat_output = gs::render_gsplat(viewpoint_camera, gaussian_model, bg_color, scaling_modifier, false);

    // Convert to RenderOutput format
    RenderOutput output;
    output.image = gsplat_output.image;
    output.viewspace_pts = gsplat_output.means2d;
    output.visibility = gsplat_output.radii > 0;
    output.radii = gsplat_output.radii;

    return output;
}
