#include "core/selective_adam.hpp"
#include "Ops.h"
#include <torch/torch.h>

namespace gs {

    torch::Tensor SelectiveAdam::step(LossClosure closure) {
        TORCH_CHECK(false, "SelectiveAdam requires visibility mask. Use step(visibility_mask) instead.");
        return {};
    }

    void SelectiveAdam::step(const torch::Tensor& visibility_mask) {
        torch::NoGradGuard no_grad;

        TORCH_CHECK(visibility_mask.dim() == 1, "visibility_mask must be 1D tensor");
        TORCH_CHECK(visibility_mask.dtype() == torch::kBool, "visibility_mask must be boolean tensor");

        const int64_t N = visibility_mask.numel();

        // Get global options
        const auto& global_options = options();

        for (auto& group : param_groups()) {
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

                // Increment step
                state.step_count++;

                // Call the fused CUDA kernel from gsplat
                gsplat::adam(
                    param,
                    param.grad(),
                    state.exp_avg,
                    state.exp_avg_sq,
                    visibility_mask, // Pass as optional
                    static_cast<float>(lr),
                    static_cast<float>(beta1),
                    static_cast<float>(beta2),
                    static_cast<float>(eps));
            }
        }
    }

} // namespace gs