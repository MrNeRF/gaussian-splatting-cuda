#include "scheduler.hpp"
#include "fused_adam.hpp"

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
} // namespace gs::training
