// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#pragma once

#include "general_utils.cuh"
#include "parameters.cuh"
#include "point_cloud.cuh"
#include "sh_utils.cuh"
#include "spatial.h"
#include <memory>
#include <string>
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
    void One_up_sh_degree();
    void Create_from_pcd(PointCloud& pcd, float spatial_lr_scale);
    void Training_setup(const OptimizationParameters& params);
    void Update_learning_rate(float lr);
    void Save_as_ply(const std::string& filename);
    void Reset_opacity();
    void Add_densification_stats(torch::Tensor& viewspace_point_tensor, torch::Tensor& update_filter);

private:
    void prune_points(const torch::Tensor& mask);
    void densification_postfix(const torch::Tensor& new_xyz,
                               const torch::Tensor& new_features_dc,
                               const torch::Tensor& new_features_rest,
                               const torch::Tensor& new_scaling,
                               const torch::Tensor& new_rotation,
                               const torch::Tensor& new_opacity);

    void densify_and_clone(torch::Tensor& grads, float grad_threshold, float scene_extent);
    void densify_and_split(torch::Tensor& grads, float grad_threshold, float scene_extent, int N = 2);
    void densify_and_prune(float max_grad, float min_opacity, float extent, float max_screen_size);

private:
    int active_sh_degree;
    int max_sh_degree;
    float _spatial_lr_scale{};
    float _percent_dense{};

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
