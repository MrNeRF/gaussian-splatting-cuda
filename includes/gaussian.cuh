#pragma once

#include "general_utils.cuh"
#include <torch/torch.h>

class GaussianModel {
public:
    GaussianModel(int sh_degree) : max_sh_degree(sh_degree),
                                   active_sh_degree(0) {

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

        // Scaling activation and its inverse
        scaling_activation = torch::exp;
        scaling_inverse_activation = torch::log;

        // Covariance activation function
        covariance_activation = [](const torch::Tensor& scaling, const torch::Tensor& scaling_modifier, const torch::Tensor& rotation) {
            auto L = build_scaling_rotation(scaling_modifier * scaling, rotation);
            auto actual_covariance = torch::mm(L, L.transpose(1, 2));
            auto symm = strip_symmetric(actual_covariance);
            return symm;
        };

        // Opacity activation and its inverse
        opacity_activation = torch::sigmoid;
        inverse_opacity_activation = inverse_sigmoid;

        // Rotation activation function
        rotation_activation = torch::nn::functional::normalize;
    }

public:
    int active_sh_degree;
    int max_sh_degree;

    torch::Tensor _xyz;
    torch::Tensor _features_dc;
    torch::Tensor _features_rest;
    torch::Tensor _scaling;
    torch::Tensor _rotation;
    torch::Tensor _opacity;
    torch::Tensor _max_radii2D;
    torch::Tensor _xyz_gradient_accum;

    torch::optim::Optimizer* optimizer;

    std::function<torch::Tensor(const torch::Tensor&)> scaling_activation = torch::exp;
    std::function<torch::Tensor(const torch::Tensor&)> scaling_inverse_activation = torch::log;

    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)> covariance_activation;

    std::function<torch::Tensor(const torch::Tensor&)> opacity_activation = torch::sigmoid;
    std::function<torch::Tensor(const torch::Tensor&)> inverse_opacity_activation = inverse_sigmoid;

    // Declare function that can hold normalize function
    std::function<torch::Tensor(const torch::Tensor&, torch::nn::functional::NormalizeFuncOptions)> rotation_activation;
};
