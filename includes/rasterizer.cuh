// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include "rasterize_points.cuh"
#include <torch/torch.h>

struct GaussianRasterizationSettings {
    int image_height;
    int image_width;
    float tanfovx;
    float tanfovy;
    torch::Tensor bg;
    float scale_modifier;
    torch::Tensor viewmatrix;
    torch::Tensor projmatrix;
    int sh_degree;
    torch::Tensor camera_center;
    bool prefiltered;
};

torch::Tensor rasterize_gaussians(torch::Tensor means3D,
                                  torch::Tensor means2D,
                                  torch::Tensor sh,
                                  torch::Tensor colors_precomp,
                                  torch::Tensor opacities,
                                  torch::Tensor scales,
                                  torch::Tensor rotations,
                                  torch::Tensor cov3Ds_precomp,
                                  GaussianRasterizationSettings raster_settings);

class _RasterizeGaussians : public torch::autograd::Function<_RasterizeGaussians> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor means3D,
                                 torch::Tensor means2D,
                                 torch::Tensor sh,
                                 torch::Tensor colors_precomp,
                                 torch::Tensor opacities,
                                 torch::Tensor scales,
                                 torch::Tensor rotations,
                                 torch::Tensor cov3Ds_precomp,
                                 GaussianRasterizationSettings raster_settings) {

        auto [num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer] = RasterizeGaussiansCUDA(
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.camera_center,
            raster_settings.prefiltered,
            false);

        ctx->save_for_backward({colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer});
        // TODO: Clean up. Too much data saved.
        ctx->saved_data["num_rendered"] = num_rendered;
        ctx->saved_data["background"] = raster_settings.bg;
        ctx->saved_data["scale_modifier"] = raster_settings.scale_modifier;
        ctx->saved_data["viewmatrix"] = raster_settings.viewmatrix;
        ctx->saved_data["projmatrix"] = raster_settings.projmatrix;
        ctx->saved_data["tanfovx"] = raster_settings.tanfovx;
        ctx->saved_data["tanfovy"] = raster_settings.tanfovy;
        ctx->saved_data["image_height"] = raster_settings.image_height;
        ctx->saved_data["image_width"] = raster_settings.image_width;
        ctx->saved_data["sh_degree"] = raster_settings.sh_degree;
        ctx->saved_data["camera_center"] = raster_settings.camera_center;
        ctx->saved_data["prefiltered"] = raster_settings.prefiltered;
        // TODO: return {color, radii};
        return torch::zeros({1});
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs) {
        auto grad_out_color = grad_outputs[0];
        auto grad_out_radii = grad_outputs[1];

        auto num_rendered = ctx->saved_data["num_rendered"].to<int>();
        auto saved = ctx->get_saved_variables();
        auto colors_precomp = saved[0];
        auto means3D = saved[1];
        auto scales = saved[2];
        auto rotations = saved[3];
        auto cov3Ds_precomp = saved[4];
        auto radii = saved[5];
        auto sh = saved[6];
        auto geomBuffer = saved[7];
        auto binningBuffer = saved[8];
        auto imgBuffer = saved[9];

        auto [grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations] = RasterizeGaussiansBackwardCUDA(
            ctx->saved_data["background"].to<torch::Tensor>(),
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            ctx->saved_data["scale_modifier"].to<float>(),
            cov3Ds_precomp,
            ctx->saved_data["viewmatrix"].to<torch::Tensor>(),
            ctx->saved_data["projmatrix"].to<torch::Tensor>(),
            ctx->saved_data["tanfovx"].to<float>(),
            ctx->saved_data["tanfovy"].to<float>(),
            grad_out_color,
            sh,
            ctx->saved_data["sh_degree"].to<int>(),
            ctx->saved_data["camera_center"].to<torch::Tensor>(),
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            false);

        return {grad_means3D, grad_means2D, grad_sh, grad_colors_precomp, grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp, torch::Tensor(), torch::Tensor()};
    }
};

class GaussianRasterizer : torch::nn::Module {
public:
    GaussianRasterizer(GaussianRasterizationSettings raster_settings) : raster_settings_(raster_settings) {}

    torch::Tensor mark_visible(torch::Tensor positions) {
        torch::NoGradGuard no_grad;
        auto visible = markVisible(
            positions,
            raster_settings_.viewmatrix,
            raster_settings_.projmatrix);

        return visible;
    }

    torch::Tensor forward(torch::Tensor means3D,
                          torch::Tensor means2D,
                          torch::Tensor opacities,
                          torch::Tensor shs = torch::Tensor(),
                          torch::Tensor colors_precomp = torch::Tensor(),
                          torch::Tensor scales = torch::Tensor(),
                          torch::Tensor rotations = torch::Tensor(),
                          torch::Tensor cov3D_precomp = torch::Tensor()) {

        if ((shs.defined() && colors_precomp.defined()) || (!shs.defined() && !colors_precomp.defined())) {
            throw std::invalid_argument("Please provide exactly one of either SHs or precomputed colors!");
        }

        if (((scales.defined() || rotations.defined()) && cov3D_precomp.defined()) ||
            (!scales.defined() && !rotations.defined() && !cov3D_precomp.defined())) {
            throw std::invalid_argument("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!");
        }

        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings_);
    }

private:
    GaussianRasterizationSettings raster_settings_;
};
