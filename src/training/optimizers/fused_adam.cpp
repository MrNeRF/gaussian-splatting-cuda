/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fused_adam.hpp"
#include "adam_api.h"

// TODO: This is just a gimmick for the bounty. I don't think it should be integrated into the main codebase.
// TODO: Removing the SH step skipping also means that the custom zero_grad() method is no longer needed.
// TODO: All skipping conditions assume that iteration count starts at 1 (which is it currently does).
// Between iteration 1000 and 25000, we can skip every second step for higher degree SH coefficients.
// This does push the bounty benchmark below 20 minutes, but I don't really like the practical implications.
// It also causes a *very* small drop in quality metrics and robustness. Thus, I disable it by default.
#define SKIP_SH_STEPS false

namespace gs::training {
    torch::Tensor FusedAdam::step(LossClosure closure) {
        TORCH_CHECK(false, "FusedAdam does not support closures.");
        return {};
    }

    void FusedAdam::step(int iteration) {
        torch::NoGradGuard no_grad;

        // Get global options
        const auto& global_options = options();

        int i = 0; // HACK: counter to track what Gaussian parameter we are on
        for (auto& group : param_groups()) {
            ++i;

            // For each group, check if it has specific options
            double lr = global_options.lr();
            double eps = global_options.eps();
            auto [beta1, beta2] = global_options.betas();

            // If the group has its own options, use those
            if (group.has_options()) {
                if (auto* group_opts = dynamic_cast<const Options*>(&group.options())) {
                    lr = group_opts->lr();
                    eps = group_opts->eps();
                    std::tie(beta1, beta2) = group_opts->betas();
                }
            }

            for (auto& param : group.params()) {
                if (!param.grad().defined()) {
                    continue;
                }

                // Lazy state initialization
                auto state_ptr = state_.find(param.unsafeGetTensorImpl());
                if (state_ptr == state_.end()) {
                    auto new_state = std::make_unique<AdamParamState>();
                    new_state->step_count = 0;
                    new_state->exp_avg = torch::zeros_like(param, torch::MemoryFormat::Preserve);
                    new_state->exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve);

                    state_[param.unsafeGetTensorImpl()] = std::move(new_state);
                    state_ptr = state_.find(param.unsafeGetTensorImpl());
                }

                auto& state = static_cast<AdamParamState&>(*state_ptr->second);

                // Increment step
                state.step_count++;

                // Higher degree SH coefficients are not used in the first 1000 iterations so this is a free speed up
                if (i == 3 && iteration <= 1000)
                    continue;

                if constexpr (SKIP_SH_STEPS) {
                    // Skip every second step during training except for the last 5000 iterations
                    if (i == 3 && (iteration % 2 != 0 && iteration <= 25000))
                        continue;
                }

                auto bias_correction1_rcp = 1.0 / (1.0 - std::pow(beta1, state.step_count));
                auto bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(beta2, state.step_count));

                // Call the fused CUDA kernel from fastgs
                fast_gs::optimizer::adam_step_wrapper(
                    param,
                    state.exp_avg,
                    state.exp_avg_sq,
                    param.grad(),
                    static_cast<float>(lr),
                    static_cast<float>(beta1),
                    static_cast<float>(beta2),
                    static_cast<float>(eps),
                    static_cast<float>(bias_correction1_rcp),
                    static_cast<float>(bias_correction2_sqrt_rcp));
            }
        }
    }

    // Based on https://github.com/pytorch/pytorch/blob/ee343ce60ceb449da09d229db25fa9d425d85a4b/torch/csrc/api/src/optim/optimizer.cpp#L122
    void FusedAdam::zero_grad(bool set_to_none, int iteration) {
        if constexpr (SKIP_SH_STEPS) {
            int i = 0; // HACK: counter to track what Gaussian parameter we are on
            for (auto& group : param_groups()) {
                ++i;
                for (auto& p : group.params()) {
                    // We want to keep accumulating if the optimizer step was skipped
                    if (i == 3 && (iteration % 2 != 0 && iteration <= 25000))
                        continue;
                    if (p.mutable_grad().defined()) {
                        p.mutable_grad().detach_();
                        if (set_to_none)
                            p.mutable_grad().reset();
                        else
                            p.mutable_grad().zero_();
                    }
                }
            }
        } else
            Optimizer::zero_grad(set_to_none);
    }
} // namespace gs::training
