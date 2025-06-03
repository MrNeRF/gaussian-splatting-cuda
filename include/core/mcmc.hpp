#pragma once

#include "core/istrategy.hpp"
#include <memory>
#include <torch/torch.h>

class MCMC : public IStrategy {
public:
    MCMC() = delete;
    MCMC(SplatData&& splat_data);

    MCMC(const MCMC&) = delete;
    MCMC& operator=(const MCMC&) = delete;
    MCMC(MCMC&&) = default;
    MCMC& operator=(MCMC&&) = default;

    // IStrategy interface implementation
    void initialize(const gs::param::OptimizationParameters& optimParams) override;
    void post_backward(int iter, gs::RenderOutput& render_output) override;
    void step(int iter) override;
    SplatData& get_model() override { return _splat_data; }
    const SplatData& get_model() const override { return _splat_data; }

private:
    // Helper functions
    torch::Tensor multinomial_sample(const torch::Tensor& weights, int n, bool replacement = true);
    int relocate_gs();
    int add_new_gs();
    void inject_noise();
    void update_optimizer_for_relocate(torch::optim::Adam* optimizer,
                                       const torch::Tensor& sampled_indices,
                                       const torch::Tensor& dead_indices,
                                       int param_position);
    void update_optimizer_for_add(torch::optim::Adam* optimizer,
                                  const torch::Tensor& sampled_indices,
                                  int param_position);

    // Member variables
    std::unique_ptr<torch::optim::Adam> _optimizer;
    SplatData _splat_data;
    std::unique_ptr<gs::param::OptimizationParameters> _params;

    // MCMC specific parameters
    int _cap_max = 1000000;
    float _noise_lr = 5e5;
    int _refine_start_iter = 500;
    int _refine_stop_iter = 25000;
    int _refine_every = 100;
    float _min_opacity = 0.005f;
    bool _verbose = false;

    // State variables
    torch::Tensor _binoms;
    float _current_lr = 0.0f;
};