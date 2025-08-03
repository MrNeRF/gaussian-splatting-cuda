#include "core/selective_adam.hpp"
#include "Ops.h"
#include <torch/torch.h>
#include "kernels/tgs_adam.cuh"
#include <ATen/ATen.h>

namespace gs {

    torch::Tensor SelectiveAdam::step(LossClosure closure) {
        TORCH_CHECK(false, "SelectiveAdam requires visibility mask. Use step(visibility_mask) instead.");
        return {};
    }

    // void SelectiveAdam::step(const int iter, const torch::Tensor& visibility_mask) {
    //     torch::NoGradGuard no_grad;

    //     TORCH_CHECK(visibility_mask.dim() == 1, "visibility_mask must be 1D tensor");
    //     TORCH_CHECK(visibility_mask.dtype() == torch::kBool, "visibility_mask must be boolean tensor");

    //     const int64_t N = visibility_mask.numel();

    //     // Get global options
    //     const auto& global_options = options();

    //     // int group_id = 0;
    //     for (auto& group : param_groups()) {

    //         // group_id++;
    //         // for shN group only update every 16th iteration
    //         // want to do this a batched update
    //         // how do I take the average of group shN grad and update only after 16 steps
    //         // if (group_id == 3 && iter % 16 != 0) continue;

    //         // For each group, check if it has specific options
    //         double lr = global_options.lr();
    //         double eps = global_options.eps();
    //         auto [beta1, beta2] = global_options.betas();
    //         double weight_decay = global_options.weight_decay();
    //         bool amsgrad = global_options.amsgrad();

    //         // If the group has its own options, use those
    //         if (group.has_options()) {
    //             if (auto* group_opts = dynamic_cast<const Options*>(&group.options())) {
    //                 lr = group_opts->lr();
    //                 eps = group_opts->eps();
    //                 std::tie(beta1, beta2) = group_opts->betas();
    //             }
    //         }

    //         for (auto& param : group.params()) {

    //             if (!param.grad().defined()) {
    //                 continue;
    //             }

    //             // Check that this parameter's first dimension matches visibility mask
    //             TORCH_CHECK(param.size(0) == N,
    //                         "Parameter first dimension (", param.size(0),
    //                         ") must match visibility mask size (", N, ")");

    //             // Lazy state initialization
    //             auto state_ptr = state_.find(param.unsafeGetTensorImpl());
    //             if (state_ptr == state_.end()) {
    //                 auto new_state = std::make_unique<AdamParamState>();
    //                 new_state->step_count = 0;
    //                 new_state->exp_avg = torch::zeros_like(param, torch::MemoryFormat::Preserve);
    //                 new_state->exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve);

    //                 state_[param.unsafeGetTensorImpl()] = std::move(new_state);
    //                 state_ptr = state_.find(param.unsafeGetTensorImpl());
    //             }

    //             auto& state = static_cast<AdamParamState&>(*state_ptr->second);

    //             // Increment step
    //             state.step_count++;

    //             // Call the fused CUDA kernel from taminggs
    //             taminggs::fused_adam(
    //                 param,
    //                 param.grad(),
    //                 state.exp_avg,
    //                 state.exp_avg_sq,
    //                 /*visibility_mask*/c10::nullopt,
    //                 static_cast<float>(lr),
    //                 static_cast<float>(beta1),
    //                 static_cast<float>(beta2),
    //                 static_cast<float>(eps),
    //                 static_cast<int64_t>(iter)
    //             );

    //             // gsplat::adam(
    //             //     param,
    //             //     param.grad(),
    //             //     state.exp_avg,
    //             //     state.exp_avg_sq,
    //             //     visibility_mask, // Pass as optional
    //             //     static_cast<float>(lr),
    //             //     static_cast<float>(beta1),
    //             //     static_cast<float>(beta2),
    //             //     static_cast<float>(eps));
    //         }
    //     }
    // }

    void SelectiveAdam::step(const int iter, const torch::Tensor& visibility_mask) {
        torch::NoGradGuard no_grad;

        TORCH_CHECK(visibility_mask.dim() == 1, "visibility_mask must be 1D");
        TORCH_CHECK(visibility_mask.dtype() == torch::kBool, "visibility_mask must be bool");
        const int64_t N = visibility_mask.numel();

        // Get global options
        const auto& global_options = options();

        int group_id = 0;
        for (auto& group : param_groups()) {
            group_id++;

            // For each group, check if it has specific options
            double lr = global_options.lr();
            double eps = global_options.eps();
            auto [beta1, beta2] = global_options.betas();
            double weight_decay = global_options.weight_decay();
            bool amsgrad = global_options.amsgrad();

            // If the group has its own options, use those
            if (group.has_options()) {
                if (auto* group_opts = dynamic_cast<const Options*>(&group.options())) {
                    lr = group_opts->lr();
                    eps = group_opts->eps();
                    std::tie(beta1, beta2) = group_opts->betas();
                }
            }

            const int  acc_steps = (group_id == 3) && iter >= do_batch_update_after_ ? update_shN_after_every_ : 1;
            const bool do_update_now = (acc_steps == 1) ? true : ((iter + 1) % acc_steps == 0);

            for (auto& param : group.params()) {

                if (!param.grad().defined()) {
                    continue;
                }

                // Check that this parameter's first dimension matches visibility mask
                TORCH_CHECK(param.size(0) == N,
                            "Parameter first dimension (", param.size(0),
                            ") must match visibility mask size (", N, ")");

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

                if (acc_steps > 1) {

                    if (!state.grad_accum.defined()) {
                        state.grad_accum = torch::zeros_like(param.grad(), torch::MemoryFormat::Preserve);
                        state.micro_count = 0;
                    }

                    state.grad_accum.add_(param.grad());
                    state.micro_count++;

                    if (!do_update_now) continue;

                    const float inv = 1.0f / static_cast<float>(state.micro_count);
                    torch::Tensor avg_grad = state.grad_accum.mul(inv);

                    // Increment step_count only on macro-step (bias-correction correctness)
                    state.step_count++;

                    taminggs::fused_adam(
                        param,
                        avg_grad,
                        state.exp_avg,
                        state.exp_avg_sq,
                        use_visibility_mask_ ? c10::optional<at::Tensor>(visibility_mask) : c10::nullopt,
                        static_cast<float>(lr),
                        static_cast<float>(beta1),
                        static_cast<float>(beta2),
                        static_cast<float>(eps),
                        static_cast<int64_t>(state.step_count)
                    );

                    state.grad_accum.zero_();
                    state.micro_count = 0;

                } else {

                    state.step_count++;

                    taminggs::fused_adam(
                        param,
                        param.grad(),
                        state.exp_avg,
                        state.exp_avg_sq,
                        use_visibility_mask_ ? c10::optional<at::Tensor>(visibility_mask) : c10::nullopt,
                        static_cast<float>(lr),
                        static_cast<float>(beta1),
                        static_cast<float>(beta2),
                        static_cast<float>(eps),
                        static_cast<int64_t>(state.step_count)
                    );

                }
            }
        }
    }

} // namespace gs