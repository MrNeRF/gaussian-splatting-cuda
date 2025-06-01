// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#pragma once

#include "core/istrategy.hpp"
#include "core/scene_info.hpp"
#include "core/splat_data.hpp"
#include <memory>
#include <torch/torch.h>

class InriaADC : public IStrategy {
public:
    InriaADC() = delete;

    // Updated constructor to take PointCloud directly
    InriaADC(int sh_degree, SplatData&& splat_data);

    InriaADC(const InriaADC&) = delete;
    InriaADC& operator=(const InriaADC&) = delete;
    InriaADC(InriaADC&&) = default;
    InriaADC& operator=(InriaADC&&) = default;

    // IStrategy interface implementation
    void initialize(const gs::param::OptimizationParameters& params) override;
    void post_backward(int iter, RenderOutput& render_output) override;
    void step(int iter) override;
    SplatData& get_model() override { return _splat_data; }
    const SplatData& get_model() const override { return _splat_data; }

private:
    struct Expon_lr_func {
        float lr_init;
        float lr_final;
        float lr_delay_steps;
        float lr_delay_mult;
        int64_t max_steps;

        Expon_lr_func(float lr_init = 0.f, float lr_final = 1.f, float lr_delay_mult = 1.f,
                      int64_t max_steps = 1000000, float lr_delay_steps = 0.f)
            : lr_init(lr_init),
              lr_final(lr_final),
              lr_delay_mult(lr_delay_mult),
              max_steps(max_steps),
              lr_delay_steps(lr_delay_steps) {}

        float operator()(int64_t step) const;
    };

    void Update_learning_rate(float iteration);
    void Reset_opacity();
    void Add_densification_stats(torch::Tensor& viewspace_point_tensor, torch::Tensor& update_filter);
    void Densify_and_prune(float max_grad, float min_opacity);
    void densify_and_clone(torch::Tensor& grads, float grad_threshold);
    void densify_and_split(torch::Tensor& grads, float grad_threshold, float min_opacity);
    void prune_points(torch::Tensor mask);
    void densification_postfix(torch::Tensor& new_xyz,
                               torch::Tensor& new_features_dc,
                               torch::Tensor& new_features_rest,
                               torch::Tensor& new_scaling,
                               torch::Tensor& new_rotation,
                               torch::Tensor& new_opacity);

    std::unique_ptr<torch::optim::Adam> _optimizer;
    SplatData _splat_data;
    float _percent_dense = 0.f;
    std::unique_ptr<gs::param::OptimizationParameters> _params;
    Expon_lr_func _xyz_scheduler_args;
    torch::Tensor _denom;
    torch::Tensor _xyz_gradient_accum;
};