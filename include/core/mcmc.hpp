#pragma once

#include "core/istrategy.hpp"
#include "core/selective_adam.hpp"
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
    bool is_refining(int iter) const override;
    void step(int iter) override;
    SplatData& get_model() override { return _splat_data; }
    const SplatData& get_model() const override { return _splat_data; }

private:
    // Simple ExponentialLR implementation since C++ API is different
    class ExponentialLR {
    public:
        ExponentialLR(torch::optim::Optimizer& optimizer, double gamma, int param_group_index = -1)
            : optimizer_(optimizer),
              gamma_(gamma),
              param_group_index_(param_group_index) {}

        void step();

    private:
        torch::optim::Optimizer& optimizer_;
        double gamma_;
        int param_group_index_;
    };

    // Helper functions
    torch::Tensor multinomial_sample(const torch::Tensor& weights, int n, bool replacement = true);
    int relocate_gs();
    int add_new_gs();
    void inject_noise();
    void update_optimizer_for_relocate(torch::optim::Optimizer* optimizer,
                                       const torch::Tensor& sampled_indices,
                                       const torch::Tensor& dead_indices,
                                       int param_position);

    // Member variables
    std::unique_ptr<torch::optim::Optimizer> _optimizer;
    std::unique_ptr<ExponentialLR> _scheduler;
    SplatData _splat_data;
    std::unique_ptr<const gs::param::OptimizationParameters> _params;

    // MCMC specific parameters
    const float _noise_lr = 5e5;

    // State variables
    torch::Tensor _binoms;

    // SelectiveAdam support
    torch::Tensor _last_visibility_mask;
};