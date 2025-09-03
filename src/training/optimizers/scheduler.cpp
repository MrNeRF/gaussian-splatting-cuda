/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scheduler.hpp"
#include "fused_adam.hpp"
#include <cmath>

namespace gs::training {
    void ExponentialLR::step() {
        if (param_group_index_ >= 0) {
            auto& group = optimizer_.param_groups()[param_group_index_];

            auto* fused_adam_options = static_cast<FusedAdam::Options*>(&group.options());
            double current_lr = fused_adam_options->lr();
            fused_adam_options->lr(current_lr * gamma_);
        } else {
            // Update all param groups
            for (auto& group : optimizer_.param_groups()) {
                auto* fused_adam_options = static_cast<FusedAdam::Options*>(&group.options());
                double current_lr = fused_adam_options->lr();
                fused_adam_options->lr(current_lr * gamma_);
            }
        }
    }

    void WarmupExponentialLR::step() {
        current_step_++;

        auto update_group = [this](int group_idx) {
            auto& group = optimizer_.param_groups()[group_idx];
            double initial_lr = initial_lrs_[group_idx];
            double new_lr;

            if (current_step_ <= warmup_steps_) {
                // Linear warmup from start_factor to 1.0
                double progress = static_cast<double>(current_step_) / warmup_steps_;
                double factor = warmup_start_factor_ + (1.0 - warmup_start_factor_) * progress;
                new_lr = initial_lr * factor;
            } else {
                // Exponential decay after warmup
                int decay_steps = current_step_ - warmup_steps_;
                new_lr = initial_lr * std::pow(gamma_, decay_steps);
            }

            // Try FusedAdam first, then regular Adam
            if (auto* fused_options = dynamic_cast<FusedAdam::Options*>(&group.options())) {
                fused_options->lr(new_lr);
            } else if (auto* adam_options = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
                adam_options->lr(new_lr);
            }
        };

        if (param_group_index_ >= 0) {
            // Update specific param group
            update_group(param_group_index_);
        } else {
            // Update all param groups
            for (size_t i = 0; i < optimizer_.param_groups().size(); ++i) {
                update_group(i);
            }
        }
    }
} // namespace gs::training
