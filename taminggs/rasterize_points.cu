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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>

#include "config.h"
#include "rasterizer.h"
#include "adam.h"
#include "rasterize_points.h"

namespace taminggs {

    std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
        auto lambda = [&t](size_t N) {
            t.resize_({(long long)N});
            return reinterpret_cast<char*>(t.contiguous().data_ptr());
        };
        return lambda;
    }

    std::function<int*(size_t N)> resizeIntFunctional(torch::Tensor& t) {
        auto lambda = [&t](size_t N) {
            t.resize_({(long long)N});
            return t.contiguous().data_ptr<int>();
        };
        return lambda;
    }

    std::function<float*(size_t N)> resizeFloatFunctional(torch::Tensor& t) {
        auto lambda = [&t](size_t N) {
            t.resize_({(long long)N});
            return t.contiguous().data_ptr<float>();
        };
        return lambda;
    }

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
        const torch::Tensor& pixel_weights)
    {
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }
    
    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;

    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor out_color = torch::full({NUM_CHAFFELS, H, W}, 0.0, float_opts);
    torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
    
    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    torch::Tensor sampleBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
    std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);
    
    torch::Tensor xy_d = torch::zeros({P, 2}, means3D.options().dtype(torch::kFloat32));
    torch::Tensor depths_d = torch::zeros({P}, means3D.options().dtype(torch::kFloat32));
    torch::Tensor radii_d = torch::zeros({P}, means3D.options().dtype(torch::kInt32));

    torch::Tensor countBuffer = torch::empty({0}, int_opts);
    torch::Tensor offsetBuffer = torch::empty({0}, int_opts);
    torch::Tensor listBuffer = torch::empty({0}, int_opts);
    torch::Tensor listBufferRender = torch::empty({0}, float_opts);
    torch::Tensor listBufferDistance = torch::empty({0}, float_opts);
    std::function<int*(size_t)> listFunc = resizeIntFunctional(listBuffer);
    std::function<float*(size_t)> listFuncRender = resizeFloatFunctional(listBufferRender);
    std::function<float*(size_t)> listFuncDistance = resizeFloatFunctional(listBufferDistance);

    float* weight_ptr = nullptr;
    float* accum_weight_ptr = nullptr;
    int* reverse_count_ptr = nullptr;
    float* blending_weight_ptr = nullptr;
    float* dist_accum_ptr = nullptr;

    torch::Tensor accumWeightsBuffer = torch::empty({0}, float_opts);
    torch::Tensor reverseCount = torch::empty({0}, int_opts);
    torch::Tensor blendingWeights = torch::empty({0}, float_opts);
    torch::Tensor distAccum = torch::empty({0}, float_opts);
    if(pixel_weights.size(0) != 0)
    {
        countBuffer = torch::full({H, W}, 0, int_opts);
        offsetBuffer = torch::full({H, W}, 0, int_opts);

        if(pixel_weights.size(0) > 1)
        {
            weight_ptr = pixel_weights.contiguous().data<float>();
            accumWeightsBuffer = torch::full({P}, 0, float_opts);
            accum_weight_ptr = accumWeightsBuffer.contiguous().data<float>();

            reverseCount = torch::full({P}, 0, int_opts);
            reverse_count_ptr = reverseCount.contiguous().data<int>();
            blendingWeights = torch::full({P}, 0, float_opts);
            blending_weight_ptr = blendingWeights.contiguous().data<float>();
            distAccum = torch::full({P}, 0, float_opts);
            dist_accum_ptr = distAccum.contiguous().data<float>();
        }
    }
    
    int rendered = 0;
    int num_buckets = 0;
    if(P != 0)
    {
        int M = 0;
        if(sh.size(0) != 0)
        {
            M = sh.size(1);
        }

        auto tup = CudaRasterizer::Rasterizer::forward(
            (float2 *)xy_d.contiguous().data_ptr<float>(),
            depths_d.contiguous().data_ptr<float>(),
            radii_d.contiguous().data_ptr<int>(),
            geomFunc,
            binningFunc,
            imgFunc,
            sampleFunc,
            listFunc,
            listFuncRender,
            listFuncDistance,
            countBuffer.contiguous().data_ptr<int>(),
            offsetBuffer.contiguous().data_ptr<int>(),
            P, degree, M,
            background.contiguous().data<float>(),
            W, H,
            means3D.contiguous().data<float>(),
            dc.contiguous().data_ptr<float>(),
            sh.contiguous().data_ptr<float>(),
            colors.contiguous().data<float>(), 
            opacity.contiguous().data<float>(), 
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data<float>(), 
            viewmatrix.contiguous().data<float>(), 
            projmatrix.contiguous().data<float>(),
            campos.contiguous().data<float>(),
            tan_fovx,
            tan_fovy,
            prefiltered,
            out_color.contiguous().data<float>(),
            radii.contiguous().data<int>(),
            debug,
            weight_ptr,
            accum_weight_ptr,
            reverse_count_ptr,
            blending_weight_ptr,
            dist_accum_ptr);

            rendered = std::get<0>(tup);
            num_buckets = std::get<1>(tup);
    }
    return std::make_tuple(rendered, num_buckets, out_color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, offsetBuffer, listBuffer, listBufferRender, listBufferDistance, xy_d, depths_d, radii_d, accumWeightsBuffer, reverseCount, blendingWeights, distAccum);
    }

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
        const bool debug) 
    {
    const int P = means3D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);
    
    int M = 0;
    if(sh.size(0) != 0)
    {	
        M = sh.size(1);
    }

    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHAFFELS}, means3D.options());
    torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
    torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_ddc = torch::zeros({P, 1, 3}, means3D.options());
    torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
    torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
    
    if(P != 0)
    {  
        CudaRasterizer::Rasterizer::backward(P, degree, M, R, B,
        background.contiguous().data<float>(),
        W, H, 
        means3D.contiguous().data<float>(),
        dc.contiguous().data<float>(),
        sh.contiguous().data<float>(),
        colors.contiguous().data<float>(),
        scales.data_ptr<float>(),
        scale_modifier,
        rotations.data_ptr<float>(),
        cov3D_precomp.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(),
        campos.contiguous().data<float>(),
        tan_fovx,
        tan_fovy,
        radii.contiguous().data<int>(),
        reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
        dL_dout_color.contiguous().data<float>(),
        dL_dmeans2D.contiguous().data<float>(),
        dL_dconic.contiguous().data<float>(),  
        dL_dopacity.contiguous().data<float>(),
        dL_dcolors.contiguous().data<float>(),
        dL_dmeans3D.contiguous().data<float>(),
        dL_dcov3D.contiguous().data<float>(),
        dL_ddc.contiguous().data<float>(),
        dL_dsh.contiguous().data<float>(),
        dL_dscales.contiguous().data<float>(),
        dL_drotations.contiguous().data<float>(),
        debug);
    }

    return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_ddc, dL_dsh, dL_dscales, dL_drotations);
    }

    torch::Tensor markVisible(
            torch::Tensor& means3D,
            torch::Tensor& viewmatrix,
            torch::Tensor& projmatrix)
    { 
    const int P = means3D.size(0);
    
    torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
    
    if(P != 0)
    {
        CudaRasterizer::Rasterizer::markVisible(P,
            means3D.contiguous().data<float>(),
            viewmatrix.contiguous().data<float>(),
            projmatrix.contiguous().data<float>(),
            present.contiguous().data<bool>());
    }
    
    return present;
    }

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
        const uint32_t M
    ){
        ADAM::adamUpdate(
            param.contiguous().data<float>(),
            param_grad.contiguous().data<float>(),
            exp_avg.contiguous().data<float>(),
            exp_avg_sq.contiguous().data<float>(),
            visible.contiguous().data<bool>(),
            lr,
            b1,
            b2,
            eps,
            N,
            M);
    }

}