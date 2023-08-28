// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include "camera.cuh"
#include "gaussian.cuh"
#include "parameters.cuh"
#include "rasterizer.cuh"
#include "sh_utils.cuh"
#include <cmath>
#include <torch/torch.h>

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render(Camera& viewpoint_camera,
                                                                                     GaussianModel& gaussianModel,
                                                                                     torch::Tensor& bg_color,
                                                                                     float scaling_modifier = 1.0,
                                                                                     torch::Tensor override_color = torch::empty({})) {
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
        .sh_degree = gaussianModel.Get_active_sh_degree(),
        .camera_center = viewpoint_camera.Get_camera_center(),
        .prefiltered = false};

    GaussianRasterizer rasterizer = GaussianRasterizer(raster_settings);

    auto means3D = gaussianModel.Get_xyz();
    auto means2D = torch::zeros_like(gaussianModel.Get_xyz()).requires_grad_(true);
    auto opacity = gaussianModel.Get_opacity();

    auto scales = torch::Tensor();
    auto rotations = torch::Tensor();
    auto cov3D_precomp = torch::Tensor();

    scales = gaussianModel.Get_scaling();
    rotations = gaussianModel.Get_rotation();

    auto shs = torch::Tensor();
    torch::Tensor colors_precomp = torch::Tensor();
    // This is nonsense. Background color not used? See orginal file colors_precomp=None line 70
    shs = gaussianModel.Get_features();

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

    // Apply visibility filter to remove occluded Gaussians.
    // TODO: I think there is no real use for means2D, isn't it?
    // render, viewspace_points, visibility_filter, radii
    return {rendererd_image, means2D, radii > 0, radii};
}
