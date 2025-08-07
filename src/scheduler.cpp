#include "core/scheduler.hpp"

void ExponentialLR::step() {
    if (param_group_index_ >= 0) {
        auto& group = optimizer_.param_groups()[param_group_index_];

        // Try to cast to our custom Options first
        if (auto* adam_options = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
            double current_lr = adam_options->lr();
            adam_options->lr(current_lr * gamma_);
        } else {
            throw std::runtime_error("Invalid options passed to ExponentialLR");
        }
    } else {
        // Update all param groups
        for (auto& group : optimizer_.param_groups()) {
            if (auto* adam_options = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
                double current_lr = adam_options->lr();
                adam_options->lr(current_lr * gamma_);
            } else {
                throw std::runtime_error("Invalid options passed to ExponentialLR");
            }
        }
    }
}
