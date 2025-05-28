// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#pragma once

#include "core/exporter.hpp"
#include "core/gaussian_init.hpp"
#include "core/parameters.hpp"
#include "core/scene_info.hpp"
#include <memory>
#include <string>
#include <torch/torch.h>

class GaussianModel {

    struct Expon_lr_func {
        float lr_init;
        float lr_final;
        float lr_delay_steps;
        float lr_delay_mult;
        int64_t max_steps;
        Expon_lr_func(float lr_init = 0.f, float lr_final = 1.f, float lr_delay_mult = 1.f, int64_t max_steps = 1000000, float lr_delay_steps = 0.f)
            : lr_init(lr_init),
              lr_final(lr_final),
              lr_delay_mult(lr_delay_mult),
              max_steps(max_steps),
              lr_delay_steps(lr_delay_steps) {}

        float operator()(int64_t step) const {
            if (step < 0 || (lr_init == 0.0 && lr_final == 0.0)) {
                return 0.0;
            }
            float delay_rate;
            if (lr_delay_steps > 0. && step != 0) {
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * std::sin(0.5 * M_PI * std::clamp((float)step / lr_delay_steps, 0.f, 1.f));
            } else {
                delay_rate = 1.0;
            }
            float t = std::clamp(static_cast<float>(step) / static_cast<float>(max_steps), 0.f, 1.f);
            float log_lerp = std::exp(std::log(lr_init) * (1.f - t) + std::log(lr_final) * t);
            return delay_rate * log_lerp;
        }
    };
public:
    GaussianModel() = delete;
    GaussianModel(int sh_degree,
                  float spatial_lr_scale,
                  gauss::init::InitTensors&& init);
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
    inline torch::Tensor Get_xyz() const { return _xyz; }
    inline torch::Tensor Get_opacity() const { return torch::sigmoid(_opacity); }
    inline torch::Tensor Get_rotation() const { return torch::nn::functional::normalize(_rotation); }
    torch::Tensor Get_features() const;
    int Get_active_sh_degree() const { return _active_sh_degree; }
    torch::Tensor Get_scaling() { return torch::exp(_scaling); }
    GaussianPointCloud to_point_cloud() const;

    // Methods
    void One_up_sh_degree();
    void Training_setup(const gs::param::OptimizationParameters& params);
    void Update_learning_rate(float iteration);
    void Reset_opacity();
    void Add_densification_stats(torch::Tensor& viewspace_point_tensor, torch::Tensor& update_filter);
    void Densify_and_prune(float max_grad, float min_opacity, float extent);

public:
    // should not be public or it should maybe be pulled out here. Not sure yet
    // This is all public mostly for debugging purposes
    std::unique_ptr<torch::optim::Adam> _optimizer;
    torch::Tensor _max_radii2D;

private:
    void prune_points(torch::Tensor mask);
    void densification_postfix(torch::Tensor& new_xyz,
                               torch::Tensor& new_features_dc,
                               torch::Tensor& new_features_rest,
                               torch::Tensor& new_scaling,
                               torch::Tensor& new_rotation,
                               torch::Tensor& new_opacity);

    void densify_and_clone(torch::Tensor& grads, float grad_threshold, float scene_extent);
    void densify_and_split(torch::Tensor& grads, float grad_threshold, float scene_extent, float min_opacity);

private:
    int _active_sh_degree = 0;
    int _max_sh_degree = 0;
    float _spatial_lr_scale = 0.f;
    float _percent_dense = 0.f;

    Expon_lr_func _xyz_scheduler_args;
    torch::Tensor _denom;
    torch::Tensor _xyz;
    torch::Tensor _features_dc;
    torch::Tensor _features_rest;
    torch::Tensor _scaling;
    torch::Tensor _rotation;
    torch::Tensor _xyz_gradient_accum;
    torch::Tensor _opacity;
};
