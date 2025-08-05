#pragma once

#include "core/istrategy.hpp"
#include "core/selective_adam.hpp"
#include "core/strategy.hpp"
#include <memory>
#include <torch/torch.h>

class DefaultStrategy : public IStrategy {
public:
    DefaultStrategy() = delete;
    DefaultStrategy(gs::SplatData&& splat_data);

    DefaultStrategy(const DefaultStrategy&) = delete;
    DefaultStrategy& operator=(const DefaultStrategy&) = delete;
    DefaultStrategy(DefaultStrategy&&) = default;
    DefaultStrategy& operator=(DefaultStrategy&&) = default;

    // IStrategy interface implementation
    void initialize(const gs::param::OptimizationParameters& optimParams) override;
    void pre_backward(gs::RenderOutput& render_output) override;
    void post_backward(int iter, gs::RenderOutput& render_output) override;
    void step(int iter) override;
    bool is_refining(int iter) const override;
    gs::SplatData& get_model() override { return _splat_data; }
    const gs::SplatData& get_model() const override { return _splat_data; }

private:
    // Helper functions
    void update_state(gs::RenderOutput& render_output);
    void duplicate(const torch::Tensor is_duplicated);
    void split(const torch::Tensor is_split);
    std::tuple<int64_t, int64_t> grow_gs(int iter);
    void remove(const torch::Tensor is_prune);
    int64_t prune_gs(int iter);
    void reset_opacity();

    // Member variables
    std::unique_ptr<torch::optim::Optimizer> _optimizer;
    std::unique_ptr<ExponentialLR> _scheduler;
    gs::SplatData _splat_data;
    std::unique_ptr<const gs::param::OptimizationParameters> _params;

    // Default strategy specific parameters
    const bool _absgrad = false;
    const std::string _key_for_gradient = "means2d";

    // State variables
    torch::Tensor _grad2d;
    torch::Tensor _count;
    torch::Tensor _radii;

    // SelectiveAdam support
    torch::Tensor _last_visibility_mask;
};
