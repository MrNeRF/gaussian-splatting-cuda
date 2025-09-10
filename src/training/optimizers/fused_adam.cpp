/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fused_adam.hpp"
#include "adam_api.h"

// Skip SH steps optimization - disabled by default
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

        int i = 0;
        for (auto& group : param_groups()) {
            ++i;

            // Get group-specific options
            double lr = global_options.lr();
            double eps = global_options.eps();
            auto [beta1, beta2] = global_options.betas();

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

                // Lazy state initialization with better memory management
                auto state_ptr = state_.find(param.unsafeGetTensorImpl());
                if (state_ptr == state_.end()) {
                    auto new_state = std::make_unique<AdamParamState>();
                    new_state->step_count = 0;

                    // Use contiguous memory and same dtype/device as param
                    new_state->exp_avg = torch::zeros_like(param, param.options())
                                            .contiguous();
                    new_state->exp_avg_sq = torch::zeros_like(param, param.options())
                                               .contiguous();

                    state_[param.unsafeGetTensorImpl()] = std::move(new_state);
                    state_ptr = state_.find(param.unsafeGetTensorImpl());
                }

                auto& state = static_cast<AdamParamState&>(*state_ptr->second);
                state.step_count++;

                // Skip higher degree SH coefficients in early iterations
                if (i == 3 && iteration <= 1000)
                    continue;

                if constexpr (SKIP_SH_STEPS) {
                    // Skip every second step during training except for the last 5000 iterations
                    if (i == 3 && (iteration % 2 != 0 && iteration <= 25000))
                        continue;
                }

                auto bias_correction1_rcp = 1.0 / (1.0 - std::pow(beta1, state.step_count));
                auto bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(beta2, state.step_count));

                // Call the fused CUDA kernel - this is already memory efficient
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

    void FusedAdam::zero_grad(bool set_to_none, int iteration) {
        // ALWAYS set gradients to None to free memory immediately
        // This is crucial for preventing memory fragmentation
        set_to_none = true;  // Force this to true

        if constexpr (SKIP_SH_STEPS) {
            int i = 0;
            for (auto& group : param_groups()) {
                ++i;
                for (auto& p : group.params()) {
                    // Skip zeroing if we're accumulating gradients
                    if (i == 3 && (iteration % 2 != 0 && iteration <= 25000))
                        continue;

                    if (p.mutable_grad().defined()) {
                        // Don't detach, just reset - this frees memory immediately
                        p.mutable_grad().reset();
                    }
                }
            }
        } else {
            // Efficient gradient clearing - always use set_to_none
            for (auto& group : param_groups()) {
                for (auto& p : group.params()) {
                    if (p.mutable_grad().defined()) {
                        // Reset without detaching - frees memory immediately
                        p.mutable_grad().reset();
                    }
                }
            }
        }
    }

    void FusedAdam::compact_state() {
        // Compact optimizer state to reduce memory fragmentation
        for (auto& [key, state_ptr] : state_) {
            if (auto* adam_state = dynamic_cast<AdamParamState*>(state_ptr.get())) {
                // Make states contiguous if they're not
                if (!adam_state->exp_avg.is_contiguous()) {
                    adam_state->exp_avg = adam_state->exp_avg.contiguous();
                }
                if (!adam_state->exp_avg_sq.is_contiguous()) {
                    adam_state->exp_avg_sq = adam_state->exp_avg_sq.contiguous();
                }

                // Optional: Reallocate to exact size if there's been resizing
                // This can help with fragmentation after densification
                if (adam_state->exp_avg.defined()) {
                    auto temp = adam_state->exp_avg.clone();
                    adam_state->exp_avg = temp;
                }
                if (adam_state->exp_avg_sq.defined()) {
                    auto temp = adam_state->exp_avg_sq.clone();
                    adam_state->exp_avg_sq = temp;
                }
            }
        }
    }
} // namespace gs::training