#include "core/strategy.hpp"

namespace strategy {
    void initialize_gaussians(gs::SplatData& splat_data) {
        const auto dev = torch::kCUDA;
        splat_data.means() = splat_data.means().to(dev).set_requires_grad(true);
        splat_data.scaling_raw() = splat_data.scaling_raw().to(dev).set_requires_grad(true);
        splat_data.rotation_raw() = splat_data.rotation_raw().to(dev).set_requires_grad(true);
        splat_data.opacity_raw() = splat_data.opacity_raw().to(dev).set_requires_grad(true);
        splat_data.sh0() = splat_data.sh0().to(dev).set_requires_grad(true);
        splat_data.shN() = splat_data.shN().to(dev).set_requires_grad(true);
    }

    std::unique_ptr<torch::optim::Optimizer> create_optimizer(
        gs::SplatData& splat_data,
        const gs::param::OptimizationParameters& params) {
        if (params.selective_adam) {
            std::cout << "Using SelectiveAdam optimizer" << std::endl;

            using Options = gs::SelectiveAdam::Options;
            std::vector<torch::optim::OptimizerParamGroup> groups;

            // Create groups with proper unique_ptr<Options>
            auto add_param_group = [&groups](const torch::Tensor& param, double lr) {
                auto options = std::make_unique<Options>(lr);
                options->eps(1e-15).betas(std::make_tuple(0.9, 0.999));
                groups.emplace_back(
                    std::vector<torch::Tensor>{param},
                    std::unique_ptr<torch::optim::OptimizerOptions>(std::move(options)));
            };

            add_param_group(splat_data.means(), params.means_lr * splat_data.get_scene_scale());
            add_param_group(splat_data.sh0(), params.shs_lr);
            add_param_group(splat_data.shN(), params.shs_lr / 20.f);
            add_param_group(splat_data.scaling_raw(), params.scaling_lr);
            add_param_group(splat_data.rotation_raw(), params.rotation_lr);
            add_param_group(splat_data.opacity_raw(), params.opacity_lr);

            auto global_options = std::make_unique<Options>(0.f);
            global_options->eps(1e-15);
            return std::make_unique<gs::SelectiveAdam>(std::move(groups), std::move(global_options));
        } else {
            using torch::optim::AdamOptions;
            std::vector<torch::optim::OptimizerParamGroup> groups;

            // Calculate initial learning rate for position
            groups.emplace_back(torch::optim::OptimizerParamGroup({splat_data.means()},
                                                                  std::make_unique<AdamOptions>(params.means_lr * splat_data.get_scene_scale())));
            groups.emplace_back(torch::optim::OptimizerParamGroup({splat_data.sh0()},
                                                                  std::make_unique<AdamOptions>(params.shs_lr)));
            groups.emplace_back(torch::optim::OptimizerParamGroup({splat_data.shN()},
                                                                  std::make_unique<AdamOptions>(params.shs_lr / 20.f)));
            groups.emplace_back(torch::optim::OptimizerParamGroup({splat_data.scaling_raw()},
                                                                  std::make_unique<AdamOptions>(params.scaling_lr)));
            groups.emplace_back(torch::optim::OptimizerParamGroup({splat_data.rotation_raw()},
                                                                  std::make_unique<AdamOptions>(params.rotation_lr)));
            groups.emplace_back(torch::optim::OptimizerParamGroup({splat_data.opacity_raw()},
                                                                  std::make_unique<AdamOptions>(params.opacity_lr)));

            for (auto& g : groups)
                static_cast<AdamOptions&>(g.options()).eps(1e-15);

            return std::make_unique<torch::optim::Adam>(groups, AdamOptions(0.f).eps(1e-15));
        }
    }

    std::unique_ptr<ExponentialLR> create_scheduler(
        const gs::param::OptimizationParameters& params,
        torch::optim::Optimizer* optimizer,
        int param_group_index) {
        // Python: gamma = 0.01^(1/max_steps)
        // This means after max_steps, lr will be 0.01 * initial_lr
        const double gamma = std::pow(0.01, 1.0 / params.iterations);
        return std::make_unique<ExponentialLR>(*optimizer, gamma, param_group_index);
    }

    void update_param_with_optimizer(
        std::function<torch::Tensor(const int, const torch::Tensor)> param_fn,
        std::function<std::unique_ptr<torch::optim::OptimizerParamState>((torch::optim::OptimizerParamState&, const torch::Tensor))> optimizer_fn,
        std::unique_ptr<torch::optim::Optimizer>& optimizer,
        gs::SplatData& splat_data,
        std::vector<size_t> param_idxs) {
        std::array<torch::Tensor*, 6> params = {
            &splat_data.means(),
            &splat_data.sh0(),
            &splat_data.shN(),
            &splat_data.scaling_raw(),
            &splat_data.rotation_raw(),
            &splat_data.opacity_raw()};

        std::array<torch::Tensor, 6> new_params;

        // Collect old parameter keys and states
        std::vector<void*> old_param_keys;
        std::array<std::unique_ptr<torch::optim::OptimizerParamState>, 6> saved_states;

        for (auto i : param_idxs) {
            auto param = params[i];
            auto new_param = param_fn(i, *param);
            new_params[i] = new_param;

            auto& old_param = optimizer->param_groups()[i].params()[0];
            void* old_param_key = old_param.unsafeGetTensorImpl();
            old_param_keys.push_back(old_param_key);

            // Check if state exists
            auto state_it = optimizer->state().find(old_param_key);
            if (state_it != optimizer->state().end()) {
                // Clone the state before modifying - handle both optimizer types
                if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(state_it->second.get())) {
                    auto new_state = optimizer_fn(*adam_state, new_param);
                    saved_states[i] = std::move(new_state);
                } else if (auto* selective_adam_state = dynamic_cast<gs::SelectiveAdam::AdamParamState*>(state_it->second.get())) {
                    auto new_state = optimizer_fn(*selective_adam_state, new_param);
                    saved_states[i] = std::move(new_state);
                } else {
                    saved_states[i] = nullptr;
                }
            } else {
                saved_states[i] = nullptr;
            }
        }

        // Now remove all old states
        for (auto key : old_param_keys) {
            optimizer->state().erase(key);
        }

        // Update parameters and add new states
        for (auto i : param_idxs) {
            optimizer->param_groups()[i].params()[0] = new_params[i];

            if (saved_states[i]) {
                void* new_param_key = new_params[i].unsafeGetTensorImpl();
                optimizer->state()[new_param_key] = std::move(saved_states[i]);
            }
        }

        // Update the splat_data with new parameters
        for (auto i : param_idxs) {
            if (i == 0) {
                splat_data.means() = new_params[i];
            } else if (i == 1) {
                splat_data.sh0() = new_params[i];
            } else if (i == 2) {
                splat_data.shN() = new_params[i];
            } else if (i == 3) {
                splat_data.scaling_raw() = new_params[i];
            } else if (i == 4) {
                splat_data.rotation_raw() = new_params[i];
            } else if (i == 5) {
                splat_data.opacity_raw() = new_params[i];
            }
        }
    }

} // namespace strategy
