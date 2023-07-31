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

    // void create_from_pcd(BasicPointCloud& pcd, float spatial_lr_scale) {
    //     this->spatial_lr_scale = spatial_lr_scale;

    //     auto fused_point_cloud = torch::from_blob(pcd.points.data(), {pcd.points.size()}).to(torch::kCUDA);
    //     auto fused_color = RGB2SH(torch::from_blob(pcd.colors.data(), {pcd.colors.size()}).to(torch::kCUDA));

    //     auto features = torch::zeros({fused_color.size(0), 3, std::pow((max_sh_degree + 1), 2)}).to(torch::kCUDA);
    //     features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0,3), 0}, fused_color);
    //     features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3), torch::indexing::Slice(1)}, 0.0);

    //     std::cout << "Number of points at initialisation : " << fused_point_cloud.size(0) << std::endl;

    //     auto dist2 = torch::clamp_min(distCUDA2(torch::from_blob(pcd.points.data(), {pcd.points.size()}).to(torch::kCUDA)), 0.0000001);
    //     auto scales = torch::log(torch::sqrt(dist2)).unsqueeze(-1).repeat({1, 3});
    //     auto rots = torch::zeros({fused_point_cloud.size(0), 4}).to(torch::kCUDA);
    //     rots.index_put_({torch::indexing::Slice(), 0}, 1);

    //     auto opacities = inverse_sigmoid(0.5 * torch::ones({fused_point_cloud.size(0), 1}).to(torch::kCUDA));

    //     _xyz = torch::nn::Parameter(fused_point_cloud.requires_grad(true));
    //     _features_dc = torch::nn::Parameter(features.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).transpose(1, 2).contiguous().requires_grad(true));
    //     _features_rest = torch::nn::Parameter(features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1)}).transpose(1, 2).contiguous().requires_grad(true));
    //     _scaling = torch::nn::Parameter(scales.requires_grad(true));
    //     _rotation = torch::nn::Parameter(rots.requires_grad(true));
    //     _opacity = torch::nn::Parameter(opacities.requires_grad(true));
    //     _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA);
    // }

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

    std::function<torch::Tensor(const torch::Tensor&)> _scaling_activation = torch::exp;
    std::function<torch::Tensor(const torch::Tensor&)> _scaling_inverse_activation = torch::log;

    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)> _covariance_activation;

    std::function<torch::Tensor(const torch::Tensor&)> _opacity_activation = torch::sigmoid;
    std::function<torch::Tensor(const torch::Tensor&)> _inverse_opacity_activation = inverse_sigmoid;

    // Declare function that can hold normalize function
    std::function<torch::Tensor(const torch::Tensor&, torch::nn::functional::NormalizeFuncOptions)> _rotation_activation;
};
