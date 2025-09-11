/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "istrategy.hpp"
#include <memory>
#include <torch/torch.h>

namespace gs::training {
    class CombinedStrategy : public IStrategy {
    public:
        CombinedStrategy() = delete;

        explicit CombinedStrategy(gs::SplatData&& splat_data);

        CombinedStrategy(const CombinedStrategy&) = delete;
        CombinedStrategy& operator=(const CombinedStrategy&) = delete;
        CombinedStrategy(CombinedStrategy&&) = default;
        CombinedStrategy& operator=(CombinedStrategy&&) = default;

        void initialize(const gs::param::OptimizationParameters& optimParams) override;
        void post_backward(int iter, RenderOutput& render_output) override;
        void step(int iter) override;
        bool is_refining(int iter) const override;

        gs::SplatData& get_model() override { return _splat_data; }
        const gs::SplatData& get_model() const override { return _splat_data; }

        void remove_gaussians(const torch::Tensor& mask) override;

    private:
        // Helpers borrowed from DefaultStrategy
        void duplicate(const torch::Tensor& is_duplicated);
        void split(const torch::Tensor& is_split);
        void remove(const torch::Tensor& is_prune);
        void prune_gs(int iter);
        void grow_gs(int iter);
        void reset_opacity();

        // Helpers borrowed from MCMC
        class ExponentialLR {
        public:
            ExponentialLR(torch::optim::Optimizer& optimizer, double gamma, int param_group_index = -1)
                : optimizer_(optimizer), gamma_(gamma), param_group_index_(param_group_index) {}
            void step();

        private:
            torch::optim::Optimizer& optimizer_;
            double gamma_;
            int param_group_index_;
        };

        torch::Tensor multinomial_sample(const torch::Tensor& weights, int n, bool replacement = true);
        int relocate_gs();
        int add_new_gs();
        void inject_noise();
        void update_optimizer_for_relocate(torch::optim::Optimizer* optimizer,
                                           const torch::Tensor& sampled_indices,
                                           const torch::Tensor& dead_indices,
                                           int param_position);

    private:
        std::unique_ptr<torch::optim::Optimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        gs::SplatData _splat_data;
        std::unique_ptr<const gs::param::OptimizationParameters> _params;

        // MCMC state
        const float _noise_lr = 5e5;
        torch::Tensor _binoms;
    };
} // namespace gs::training


