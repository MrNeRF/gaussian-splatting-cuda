/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <torch/torch.h>

namespace gs::training {
    // Simple ExponentialLR implementation since C++ API is different
    class ExponentialLR {
    public:
        ExponentialLR(torch::optim::Optimizer& optimizer, double gamma, int param_group_index = -1)
            : optimizer_(optimizer),
              gamma_(gamma),
              param_group_index_(param_group_index) {
        }

        void step();

    private:
        torch::optim::Optimizer& optimizer_;
        double gamma_;
        int param_group_index_;
    };

    class WarmupExponentialLR {
    public:
        WarmupExponentialLR(
            torch::optim::Optimizer& optimizer,
            double gamma,
            int warmup_steps = 0,
            double warmup_start_factor = 1.0,
            int param_group_index = -1)
            : optimizer_(optimizer),
              gamma_(gamma),
              warmup_steps_(warmup_steps),
              warmup_start_factor_(warmup_start_factor),
              param_group_index_(param_group_index),
              current_step_(0) {
            // Store initial learning rates for all param groups
            for (const auto& group : optimizer.param_groups()) {
                auto* options = static_cast<torch::optim::AdamOptions*>(&const_cast<torch::optim::OptimizerParamGroup&>(group).options());
                initial_lrs_.push_back(options->lr());
            }
        }

        void step();

    private:
        torch::optim::Optimizer& optimizer_;
        double gamma_;
        int warmup_steps_;
        double warmup_start_factor_;
        int param_group_index_;
        int current_step_;
        std::vector<double> initial_lrs_;
    };
} // namespace gs::training
