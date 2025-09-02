/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "istrategy.hpp"
#include "optimizers/scheduler.hpp"
#include <memory>
#include <torch/torch.h>

namespace gs::training {
    // Forward declarations
    struct RenderOutput;
    class FusedAdam;

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

        void post_backward(int iter, RenderOutput& render_output) override;

        void step(int iter) override;

        bool is_refining(int iter) const override;

        gs::SplatData& get_model() override { return _splat_data; }
        const gs::SplatData& get_model() const override { return _splat_data; }

        void remove_gaussians(const torch::Tensor& mask) override;

    private:
        // Helper functions
        void duplicate(const torch::Tensor& is_duplicated);

        void split(const torch::Tensor& is_split);

        void grow_gs(int iter);

        void remove(const torch::Tensor& is_prune);

        void prune_gs(int iter);

        void reset_opacity();

        // Member variables
        std::unique_ptr<torch::optim::Optimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        gs::SplatData _splat_data;
        std::unique_ptr<const gs::param::OptimizationParameters> _params;
    };
} // namespace gs::training
