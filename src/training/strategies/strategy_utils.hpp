#pragma once

#include "istrategy.hpp"
#include "optimizers/scheduler.hpp"
#include <memory>
#include <torch/torch.h>

namespace strategy {
    void initialize_gaussians(gs::SplatData& splat_data);

    std::unique_ptr<torch::optim::Optimizer> create_optimizer(
        gs::SplatData& splat_data,
        const gs::param::OptimizationParameters& params);

    std::unique_ptr<ExponentialLR> create_scheduler(
        const gs::param::OptimizationParameters& params,
        torch::optim::Optimizer* optimizer,
        int param_group_index = -1);

    // Use explicit type alias to help MSVC
    using ParamUpdateFn = std::function<torch::Tensor(const int, const torch::Tensor)>;
    using OptimizerUpdateFn = std::function<std::unique_ptr<torch::optim::OptimizerParamState>(torch::optim::OptimizerParamState&, const torch::Tensor)>;

    void update_param_with_optimizer(
        const ParamUpdateFn& param_fn,
        const OptimizerUpdateFn& optimizer_fn,
        std::unique_ptr<torch::optim::Optimizer>& optimizer,
        gs::SplatData& splat_data,
        std::vector<size_t> param_idxs = {0, 1, 2, 3, 4, 5});

} // namespace strategy