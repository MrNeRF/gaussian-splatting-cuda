// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include "camera.cuh"
#include "gaussian.cuh"
#include "parameters.cuh"
#include "rasterizer.cuh"
#include <cmath>
#include <torch/torch.h>

torch::Tensor render(Camera& viewpoint_camera, GaussianModel& gaussianModel,
                     const PipelineParameters& params,
                     torch::Tensor& bg_color,
                     float scaling_modifier = 1.0,
                     torch::Tensor override_color = torch::empty({})) {
    // Ensure background tensor (bg_color) is on GPU!
    bg_color = bg_color.to(torch::kCUDA);

    // Set up rasterization configuration
    auto raster_settings = GaussianRasterizationSettings();
    //    GaussianRasterizationSettings raster_settings = {
    //        .image_height=static_cast<int>(viewpoint_camera._image_height),
    //        .image_width=static_cast<int>(viewpoint_camera._image_width),
    //        .tanfovx=std::tan(viewpoint_camera._fov_x * 0.5f),
    //        .tanfovy=std::tan(viewpoint_camera._fov_x * 0.5f),
    //        .bg=bg_color,
    //        .scale_modifier=scaling_modifier,
    //        .viewmatrix=viewpoint_camera.world_view_transform,
    //        .projmatrix=viewpoint_camera.full_proj_transform,
    //        .sh_degree= gaussianModel.Get_active_sh_degree(),
    //        .camera_center=viewpoint_camera.camera_center,
    //        .prefiltered=false
    //    };

    GaussianRasterizer rasterizer = GaussianRasterizer(raster_settings);

    auto means3D = gaussianModel.Get_xyz();
    auto means2D = torch::zeros_like(gaussianModel.Get_xyz()).requires_grad_(true);
    auto opacity = gaussianModel.Get_opacity();

    auto scales = torch::empty({});
    auto rotations = torch::empty({});
    auto cov3D_precomp = torch::empty({});

    if (params.compute_cov3D_python) {
        cov3D_precomp = gaussianModel.Get_covariance(scaling_modifier);
    } else {
        scales = gaussianModel.Get_scaling();
        rotations = gaussianModel.Get_rotation();
    }

    auto shs = torch::empty({});
    torch::Tensor colors_precomp = torch::empty({});
    // This is nonsense. Background color not used? See orginal file colors_precomp=None line 70
    if (params.convert_SHs_python) {
        // TODO: Camera center is not properly implemented. Need to correct this first.
        //            torch::Tensor shs_view = gaussianModel.Get_features.transpose(1, 2).view({-1, 3, std::pow(gaussianModel.Get_max_sh_degree()+1,2)});
        //            torch::Tensor dir_pp = (gaussianModel.Get_xyz - viewpoint_camera._camera_center.repeat(gaussianModel.Get_features.sizes()[0], 1));
        //            torch::Tensor dir_pp_normalized = dir_pp/dir_pp.norm(1);
        //            torch::Tensor sh2rgb = eval_sh(gaussianModel.active_sh_degree, shs_view, dir_pp_normalized);
        //            colors_precomp = torch::clamp_min(sh2rgb + 0.5, 0.0);
    } else {
        shs = gaussianModel.Get_features();
    }

    // Rasterize visible Gaussians to image, obtain their radii (on screen).
    //    std::pair<torch::Tensor, torch::Tensor> rasterize_result = rasterizer.forward(
    //        means3D,
    //        means2D,
    //        shs,
    //        colors_precomp,
    //        opacity,
    //        scales,
    //        rotations,
    //        cov3D_precomp);
    //    torch::Tensor rendered_image = rasterize_result.first;
    //    torch::Tensor radii = rasterize_result.second;

    // Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    // They will be excluded from value updates used in the splitting criteria.
    //    std::map<std::string, torch::Tensor> result;
    //    result.insert(std::pair<std::string, torch::Tensor>("render", rendered_image));
    //    result.insert(std::pair<std::string, torch::Tensor>("viewspace_points", screenspace_points));
    //    result.insert(std::pair<std::string, torch::Tensor>("visibility_filter", radii > 0));
    //    result.insert(std::pair<std::string, torch::Tensor>("radii", radii));

    //    return result;
    return torch::empty({});
}
