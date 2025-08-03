/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

namespace taminggs {
	
    std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    RasterizeGaussiansCUDA(
        const torch::Tensor& background,
        const torch::Tensor& means3D,
        const torch::Tensor& colors,
        const torch::Tensor& opacity,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const float scale_modifier,
        const torch::Tensor& cov3D_precomp,
        const torch::Tensor& viewmatrix,
        const torch::Tensor& projmatrix,
        const float tan_fovx, 
        const float tan_fovy,
        const int image_height,
        const int image_width,
        const torch::Tensor& dc,
        const torch::Tensor& sh,
        const int degree,
        const torch::Tensor& campos,
        const bool prefiltered,
        const bool debug,
        const torch::Tensor& pixel_weights);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    RasterizeGaussiansBackwardCUDA(
        const torch::Tensor& background,
        const torch::Tensor& means3D,
        const torch::Tensor& radii,
        const torch::Tensor& colors,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const float scale_modifier,
        const torch::Tensor& cov3D_precomp,
        const torch::Tensor& viewmatrix,
        const torch::Tensor& projmatrix,
        const float tan_fovx, 
        const float tan_fovy,
        const torch::Tensor& dL_dout_color,
        const torch::Tensor& dc,
        const torch::Tensor& sh,
        const int degree,
        const torch::Tensor& campos,
        const torch::Tensor& geomBuffer,
        const int R,
        const torch::Tensor& binningBuffer,
        const torch::Tensor& imageBuffer,
        const int B,
        const torch::Tensor& sampleBuffer,
        const bool debug);
            
    torch::Tensor markVisible(
            torch::Tensor& means3D,
            torch::Tensor& viewmatrix,
            torch::Tensor& projmatrix);

    torch::Tensor conv2DForward(torch::Tensor &input);

    void adamUpdate(
        torch::Tensor &param,
        torch::Tensor &param_grad,
        torch::Tensor &exp_avg,
        torch::Tensor &exp_avg_sq,
        torch::Tensor &visible,
        const float lr,
        const float b1,
        const float b2,
        const float eps,
        const uint32_t N,
        const uint32_t M);

}