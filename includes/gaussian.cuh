// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#pragma once

#include "general_utils.cuh"
#include "parameters.cuh"
#include "point_cloud.cuh"
#include "sh_utils.cuh"
#include "spatial.h"
#include <memory>
#include <torch/torch.h>

class GaussianModel : torch::nn::Module {
public:
    GaussianModel(int sh_degree) : max_sh_degree(sh_degree),
                                   active_sh_degree(0),
                                   _xyz_scheduler_args(Expon_lr_func(0.0, 1.0)) {

        // Assuming these are 1D tensors
        _xyz = torch::empty({0});
        _features_dc = torch::empty({0});
        _features_rest = torch::empty({0});
        _scaling = torch::empty({0});
        _rotation = torch::empty({0});
        _opacity = torch::empty({0});
        _max_radii2D = torch::empty({0});
        _xyz_gradient_accum = torch::empty({0});
        optimizer = nullptr;

        register_parameter("xyz", _xyz, true);
        register_parameter("features_dc", _features_dc, true);
        register_parameter("features_rest", _features_rest, true);
        register_parameter("scaling", _scaling, true);
        register_parameter("rotation", _rotation, true);
        register_parameter("opacity", _opacity, true);

        // Scaling activation and its inverse
        _scaling_activation = torch::exp;
        _scaling_inverse_activation = torch::log;

        // Covariance activation function
        _covariance_activation = [](const torch::Tensor& scaling, const torch::Tensor& scaling_modifier, const torch::Tensor& rotation) {
            auto L = build_scaling_rotation(scaling_modifier * scaling, rotation);
            auto actual_covariance = torch::mm(L, L.transpose(1, 2));
            auto symm = strip_symmetric(actual_covariance);
            return symm;
        };

        // Opacity activation and its inverse
        _opacity_activation = torch::sigmoid;
        _inverse_opacity_activation = inverse_sigmoid;

        // Rotation activation function
        _rotation_activation = torch::nn::functional::normalize;
    }

public:
    // Getters
    torch::Tensor get_xyz() {
        return _xyz;
    }

    torch::Tensor get_features() {
        auto features_dc = _features_dc;
        auto features_rest = _features_rest;
        return torch::cat({features_dc, features_rest}, 1);
    }

    torch::Tensor get_opacity() {
        return _opacity_activation(_opacity);
    }

    // torch::Tensor get_covariance(double scaling_modifier = 1.0) {
    //     return _covariance_activation(get_scaling(), scaling_modifier, _rotation);
    // }

    // Methods
    void oneupSHdegree() {
        if (active_sh_degree < max_sh_degree) {
            active_sh_degree++;
        }
    }

    void create_from_pcd(PointCloud& pcd, float spatial_lr_scale) {
        _spatial_lr_scale = spatial_lr_scale;

        torch::Tensor fused_point_cloud = torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size())}).to(torch::kCUDA);
        auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size())}).to(torch::kCUDA));

        auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((max_sh_degree + 1), 2))}).to(torch::kCUDA);
        features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}, fused_color);
        features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3), torch::indexing::Slice(1)}, 0.0);

        std::cout << "Number of points at initialisation : " << fused_point_cloud.size(0) << std::endl;

        auto dist2 = torch::clamp_min(distCUDA2(torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size())}).to(torch::kCUDA)), 0.0000001);
        auto scales = torch::log(torch::sqrt(dist2)).unsqueeze(-1).repeat({1, 3});
        auto rots = torch::zeros({fused_point_cloud.size(0), 4}).to(torch::kCUDA);
        rots.index_put_({torch::indexing::Slice(), 0}, 1);

        auto opacities = inverse_sigmoid(0.5 * torch::ones({fused_point_cloud.size(0), 1}).to(torch::kCUDA));

        _xyz = fused_point_cloud.set_requires_grad(true);
        _features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).transpose(1, 2).contiguous();
        _features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1)}).transpose(1, 2).contiguous();
        _scaling = scales;
        _rotation = rots;
        _opacity = opacities;
        _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA);
    }

    void training_setup(const OptimizationParameters params) {
        this->percent_dense = params.percent_dense;
        this->_xyz_gradient_accum = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);
        this->_denom = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);

        register_parameter("xyz", this->_xyz);
        optimizer = std::make_unique<torch::optim::Adam>(parameters(), torch::optim::AdamOptions(0.0).eps(1e-15));
        this->_xyz_scheduler_args = Expon_lr_func(params.position_lr_init * this->_spatial_lr_scale,
                                                  params.position_lr_final * this->_spatial_lr_scale,
                                                  params.position_lr_delay_mult,
                                                  params.position_lr_max_steps);
    }

public:
    int active_sh_degree;
    int max_sh_degree;
    float _spatial_lr_scale;
    float percent_dense;

    Expon_lr_func _xyz_scheduler_args;
    torch::Tensor _denom;
    torch::Tensor _xyz;
    torch::Tensor _features_dc;
    torch::Tensor _features_rest;
    torch::Tensor _scaling;
    torch::Tensor _rotation;
    torch::Tensor _opacity;
    torch::Tensor _max_radii2D;
    torch::Tensor _xyz_gradient_accum;

    std::unique_ptr<torch::optim::Adam> optimizer;

    std::function<torch::Tensor(const torch::Tensor&)> _scaling_activation = torch::exp;
    std::function<torch::Tensor(const torch::Tensor&)> _scaling_inverse_activation = torch::log;

    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)> _covariance_activation;

    std::function<torch::Tensor(const torch::Tensor&)> _opacity_activation = torch::sigmoid;
    std::function<torch::Tensor(const torch::Tensor&)> _inverse_opacity_activation = inverse_sigmoid;

    // Declare function that can hold normalize function
    std::function<torch::Tensor(const torch::Tensor&, torch::nn::functional::NormalizeFuncOptions)> _rotation_activation;
};
