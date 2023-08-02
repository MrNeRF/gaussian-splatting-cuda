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
    explicit GaussianModel(int sh_degree);

    // Copy constructor
    GaussianModel(const GaussianModel& other) = delete;

    // Copy assignment operator
    GaussianModel& operator=(const GaussianModel& other) = delete;

    // Move constructor
    GaussianModel(GaussianModel&& other) = default;

    // Move assignment operator
    GaussianModel& operator=(GaussianModel&& other) = default;

public:
    // Getters
    inline torch::Tensor get_xyz() const { return _xyz; }
    inline torch::Tensor get_opacity() const { return _opacity_activation(_opacity); }
    torch::Tensor get_features() const;
    // torch::Tensor get_covariance(double scaling_modifier = 1.0) {
    //     return _covariance_activation(get_scaling(), scaling_modifier, _rotation);
    // }

    // Methods
    void OneupSHdegree();
    void Create_from_pcd(PointCloud& pcd, float spatial_lr_scale);
    void Training_setup(const OptimizationParameters& params);
    void Update_Learning_Rate(float lr);
    void Save_As_PLY(const std::string& filename);
    void Reset_Opacity();

public:
    int active_sh_degree;
    int max_sh_degree;
    float _spatial_lr_scale{};
    float percent_dense{};

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

    std::unique_ptr<torch::optim::Adam> _optimizer;

    std::function<torch::Tensor(const torch::Tensor&)> _scaling_activation = torch::exp;
    std::function<torch::Tensor(const torch::Tensor&)> _scaling_inverse_activation = torch::log;

    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)> _covariance_activation;

    std::function<torch::Tensor(const torch::Tensor&)> _opacity_activation = torch::sigmoid;
    std::function<torch::Tensor(const torch::Tensor&)> _inverse_opacity_activation = inverse_sigmoid;

    // Declare function that can hold normalize function
    std::function<torch::Tensor(const torch::Tensor&, torch::nn::functional::NormalizeFuncOptions)> _rotation_activation;
};
