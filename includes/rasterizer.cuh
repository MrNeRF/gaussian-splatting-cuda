// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include "debug_utils.cuh"
#include "rasterize_points.cuh"

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

torch::autograd::tensor_list rasterize_gaussians(torch::Tensor means3D,
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
    static torch::autograd::tensor_list forward(torch::autograd::AutogradContext* ctx,
                                                torch::Tensor means3D,
                                                torch::Tensor means2D,
                                                torch::Tensor sh,
                                                torch::Tensor colors_precomp,
                                                torch::Tensor opacities,
                                                torch::Tensor scales,
                                                torch::Tensor rotations,
                                                torch::Tensor cov3Ds_precomp,
                                                torch::Tensor image_height,
                                                torch::Tensor image_width,
                                                torch::Tensor tanfovx,
                                                torch::Tensor tanfovy,
                                                torch::Tensor bg,
                                                torch::Tensor scale_modifier,
                                                torch::Tensor viewmatrix,
                                                torch::Tensor projmatrix,
                                                torch::Tensor sh_degree,
                                                torch::Tensor camera_center,
                                                torch::Tensor prefiltered) {

        //        ts::print_debug_info(means3D, "means3D");
        //        ts::print_debug_info(sh, "sh");
        //        ts::print_debug_info(colors_precomp, "colors_precomp");
        //        ts::print_debug_info(opacities, "opacities");
        //        ts::print_debug_info(scales, "scales");
        //        ts::print_debug_info(rotations, "rotations");
        //        ts::print_debug_info(cov3Ds_precomp, "cov3Ds_precomp");
        //        ts::print_debug_info(image_height, "image_height");
        //        ts::print_debug_info(image_width, "image_width");
        //        ts::print_debug_info(tanfovx, "tanfovx");
        //        ts::print_debug_info(tanfovy, "tanfovy");
        //        ts::print_debug_info(bg, "bg");
        //        ts::print_debug_info(scale_modifier, "scale_modifier");
        //        ts::print_debug_info(viewmatrix, "viewmatrix");
        //        ts::print_debug_info(projmatrix, "projmatrix");
        //        ts::print_debug_info(sh_degree, "sh_degree");
        //        ts::print_debug_info(camera_center, "camera_center");
        //        ts::print_debug_info(prefiltered, "prefiltered");

        int image_height_val = image_height.item<int>();
        int image_width_val = image_width.item<int>();
        float tanfovx_val = tanfovx.item<float>();
        float tanfovy_val = tanfovy.item<float>();
        float scale_modifier_val = scale_modifier.item<float>();
        int sh_degree_val = sh_degree.item<int>();
        bool prefiltered_val = prefiltered.item<bool>();

        // TODO: should it be this way? Bug?
        camera_center = camera_center.contiguous();

        auto [num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer] = RasterizeGaussiansCUDA(
            bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            scale_modifier_val,
            cov3Ds_precomp,
            viewmatrix,
            projmatrix,
            tanfovx_val,
            tanfovy_val,
            image_height_val,
            image_width_val,
            sh,
            sh_degree_val,
            camera_center,
            prefiltered_val,
            true);

        ctx->save_for_backward({colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer});
        // TODO: Clean up. Too much data saved.
        ctx->saved_data["num_rendered"] = num_rendered;
        ctx->saved_data["background"] = bg;
        ctx->saved_data["scale_modifier"] = scale_modifier_val;
        ctx->saved_data["viewmatrix"] = viewmatrix;
        ctx->saved_data["projmatrix"] = projmatrix;
        ctx->saved_data["tanfovx"] = tanfovx_val;
        ctx->saved_data["tanfovy"] = tanfovy_val;
        ctx->saved_data["image_height"] = image_height_val;
        ctx->saved_data["image_width"] = image_width_val;
        ctx->saved_data["sh_degree"] = sh_degree_val;
        ctx->saved_data["camera_center"] = camera_center;
        ctx->saved_data["prefiltered"] = prefiltered_val;
        return {color, radii};
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
            true);

        // return gradients for all inputs, 19 in total. :D
        return {grad_means3D,
                grad_means2D,
                grad_sh,
                grad_colors_precomp,
                grad_opacities,
                grad_scales,
                grad_rotations,
                grad_cov3Ds_precomp,
                torch::Tensor(), // from here placeholder, not used: #forwards args = #backwards args.
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor()};
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

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor means3D,
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

        // Check if tensors are undefined, and if so, initialize them
        torch::Device device = torch::kCUDA;
        if (!shs.defined()) {
            shs = torch::empty({0}, device);
        }
        if (!colors_precomp.defined()) {
            colors_precomp = torch::empty({0}, device);
        }
        if (!scales.defined()) {
            scales = torch::empty({0}, device);
        }
        if (!rotations.defined()) {
            rotations = torch::empty({0}, device);
        }
        if (!cov3D_precomp.defined()) {
            cov3D_precomp = torch::empty({0}, device);
        }

        // match datalayout to python implementation
        //        means3D = means3D.transpose(0, 1);

        auto result = rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings_);

        return {result[0], result[1]};
    }

private:
    GaussianRasterizationSettings raster_settings_;
};
