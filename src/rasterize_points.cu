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

#include "rasterize_points.cuh"
#include <config.h>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <rasterizer.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <torch/extension.h>
#include <tuple>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

void print_tensor_info(torch::Tensor tensor, std::string tensor_name) {
    std::cout << "------ " << tensor_name << " ------\n";
    std::cout << "Device: " << (tensor.device().is_cuda() ? "CUDA" : "CPU") << "\n";
    std::cout << "Data type: " << tensor.dtype() << "\n";
    std::cout << "Is contiguous: " << (tensor.is_contiguous() ? "True" : "False") << "\n";
    std::cout << "Is pinned: " << (tensor.is_pinned() ? "True" : "False") << "\n";
    std::cout << "Size: " << tensor.sizes() << "\n";
    std::cout << "Strides: " << tensor.strides() << "\n";
    if (tensor.device().is_cuda()) {
        auto ptr = tensor.data_ptr();
        std::cout << "Data pointer: " << (ptr != nullptr ? "Not null" : "Null") << "\n";
    }
    std::cout << "----------------------------\n";
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug) {
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }

    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;

    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
    torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

    int rendered = 0;
    if (P != 0) {
        int M = 0;
        if (sh.size(0) != 0) {
            M = sh.size(1);
        }
        //        print_tensor_info(background, "background");
        //        print_tensor_info(means3D, "means3D");
        //        print_tensor_info(colors, "colors");
        //        print_tensor_info(opacity, "opacity");
        //        print_tensor_info(scales, "scales");
        //        print_tensor_info(rotations, "rotations");
        //        print_tensor_info(cov3D_precomp, "cov3D_precomp");
        //        print_tensor_info(viewmatrix, "viewmatrix");
        //        print_tensor_info(projmatrix, "projmatrix");
        //        print_tensor_info(sh, "sh");
        //        print_tensor_info(campos, "campos");
        //        print_tensor_info(out_color, "out_color");
        //        print_tensor_info(radii, "radii");

        //        if (!background.data_ptr<float>()) {
        //            std::cout << "Null data pointer: background"
        //                      << "\n";
        //        }
        //        if (!means3D.data_ptr<float>()) {
        //            std::cout << "Null data pointer: means3D"
        //                      << "\n";
        //        }
        //        if (!colors.data_ptr<float>()) {
        //            std::cout << "Null data pointer: colors"
        //                      << "\n";
        //        }
        //        if (!opacity.data_ptr<float>()) {
        //            std::cout << "Null data pointer: opacity"
        //                      << "\n";
        //        }
        //        if (!scales.data_ptr<float>()) {
        //            std::cout << "Null data pointer: scales"
        //                      << "\n";
        //        }
        //        if (!rotations.data_ptr<float>()) {
        //            std::cout << "Null data pointer: rotations"
        //                      << "\n";
        //        }
        //        if (!cov3D_precomp.data_ptr<float>()) {
        //            std::cout << "Null data pointer: cov3D_precomp"
        //                      << "\n";
        //        }
        //        if (!viewmatrix.data_ptr<float>()) {
        //            std::cout << "Null data pointer: viewmatrix"
        //                      << "\n";
        //        }
        //        if (!projmatrix.data_ptr<float>()) {
        //            std::cout << "Null data pointer: projmatrix"
        //                      << "\n";
        //        }
        //        if (!sh.data_ptr<float>()) {
        //            std::cout << "Null data pointer: sh"
        //                      << "\n";
        //        }
        //        if (!campos.data_ptr<float>()) {
        //            std::cout << "Null data pointer: campos"
        //                      << "\n";
        //        }
        //        if (!out_color.data_ptr<float>()) {
        //            std::cout << "Null data pointer: out_color"
        //                      << "\n";
        //        }
        //        if (!radii.data_ptr<int>()) {
        //            std::cout << "Null data pointer: radii"
        //                      << "\n";
        //        }

        rendered = CudaRasterizer::Rasterizer::forward(
            geomFunc,
            binningFunc,
            imgFunc,
            P, degree, M,
            background.contiguous().data_ptr<float>(),
            W, H,
            means3D.contiguous().data_ptr<float>(),
            sh.contiguous().data_ptr<float>(),
            colors.contiguous().data_ptr<float>(),
            opacity.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            prefiltered,
            out_color.contiguous().data_ptr<float>(),
            radii.contiguous().data_ptr<int>(),
            debug);
    }
    return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool debug) {
    const int P = means3D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);

    int M = 0;
    if (sh.size(0) != 0) {
        M = sh.size(1);
    }

    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
    torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
    torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
    torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

    if (P != 0) {
        CudaRasterizer::Rasterizer::backward(P, degree, M, R,
                                             background.contiguous().data_ptr<float>(),
                                             W, H,
                                             means3D.contiguous().data_ptr<float>(),
                                             sh.contiguous().data_ptr<float>(),
                                             colors.contiguous().data_ptr<float>(),
                                             scales.data_ptr<float>(),
                                             scale_modifier,
                                             rotations.data_ptr<float>(),
                                             cov3D_precomp.contiguous().data_ptr<float>(),
                                             viewmatrix.contiguous().data_ptr<float>(),
                                             projmatrix.contiguous().data_ptr<float>(),
                                             campos.contiguous().data_ptr<float>(),
                                             tan_fovx,
                                             tan_fovy,
                                             radii.contiguous().data_ptr<int>(),
                                             reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
                                             reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
                                             reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
                                             dL_dout_color.contiguous().data_ptr<float>(),
                                             dL_dmeans2D.contiguous().data_ptr<float>(),
                                             dL_dconic.contiguous().data_ptr<float>(),
                                             dL_dopacity.contiguous().data_ptr<float>(),
                                             dL_dcolors.contiguous().data_ptr<float>(),
                                             dL_dmeans3D.contiguous().data_ptr<float>(),
                                             dL_dcov3D.contiguous().data_ptr<float>(),
                                             dL_dsh.contiguous().data_ptr<float>(),
                                             dL_dscales.contiguous().data_ptr<float>(),
                                             dL_drotations.contiguous().data_ptr<float>(),
                                             debug);
    }

    return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix) {
    const int P = means3D.size(0);

    torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

    if (P != 0) {
        CudaRasterizer::Rasterizer::markVisible(P,
                                                means3D.contiguous().data_ptr<float>(),
                                                viewmatrix.contiguous().data_ptr<float>(),
                                                projmatrix.contiguous().data_ptr<float>(),
                                                present.contiguous().data_ptr<bool>());
    }

    return present;
}