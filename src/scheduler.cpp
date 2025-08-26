#include "core/scheduler.hpp"
#include "core/fused_adam.hpp"

void ExponentialLR::step() {
    if (param_group_index_ >= 0) {
        auto& group = optimizer_.param_groups()[param_group_index_];

        auto* fused_adam_options = static_cast<gs::FusedAdam::Options*>(&group.options());
        double current_lr = fused_adam_options->lr();
        fused_adam_options->lr(current_lr * gamma_);
    } else {
        // Update all param groups
        for (auto& group : optimizer_.param_groups()) {
            auto* fused_adam_options = static_cast<gs::FusedAdam::Options*>(&group.options());
            double current_lr = fused_adam_options->lr();
            fused_adam_options->lr(current_lr * gamma_);
        }
    }
}