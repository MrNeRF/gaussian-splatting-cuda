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
                                                                                     const PipelineParameters& params,
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

    if (params.compute_cov3D_python) {
        cov3D_precomp = gaussianModel.Get_covariance(scaling_modifier);
    } else {
        scales = gaussianModel.Get_scaling();
        rotations = gaussianModel.Get_rotation();
    }

    auto shs = torch::Tensor();
    torch::Tensor colors_precomp = torch::Tensor();
    // This is nonsense. Background color not used? See orginal file colors_precomp=None line 70
    if (params.convert_SHs_python) {
        torch::Tensor shs_view = gaussianModel.Get_features().transpose(1, 2).view({-1, 3, static_cast<long>(std::pow(gaussianModel.Get_max_sh_degree() + 1, 2))});
        torch::Tensor dir_pp = (gaussianModel.Get_xyz() - viewpoint_camera.Get_camera_center().repeat(gaussianModel.Get_features().sizes()[0], 1));
        torch::Tensor dir_pp_normalized = dir_pp / dir_pp.norm(1);
        torch::Tensor sh2rgb = Eval_sh(gaussianModel.Get_active_sh_degree(), shs_view, dir_pp_normalized);
        colors_precomp = torch::clamp_min(sh2rgb + 0.5, 0.0);
    } else {
        shs = gaussianModel.Get_features();
    }

    torch::cuda::synchronize();
    // Rasterize visible Gaussians to image, obtain their radii (on screen).
    auto [rendererd_image, radii] = rasterizer.forward(
        means3D,
        means2D,
        shs,
        colors_precomp,
        opacity,
        scales,
        rotations,
        cov3D_precomp);

    // Apply visibility filter to remove occluded Gaussians.
    // TODO: I think there is no real use for means2D, isn't it?
    // render, viewspace_points, visibility_filter, radii
    return {rendererd_image, means2D, radii > 0, radii};
}
