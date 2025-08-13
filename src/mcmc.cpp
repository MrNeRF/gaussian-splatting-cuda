#include "core/mcmc.hpp"
#include "Ops.h"
#include "core/fused_adam.hpp"
#include "core/parameters.hpp"
#include "core/rasterizer.hpp"
#include <iostream>
#include <random>

<<<<<<< HEAD
#ifdef _WIN32
#include <c10/cuda/CUDACachingAllocator.h> //required for emptyCache
#endif

=======
>>>>>>> parent of ea456f4 (Fix mcmc memory usage in windows (#274))
void MCMC::ExponentialLR::step() {
    if (param_group_index_ >= 0) {
        auto& group = optimizer_.param_groups()[param_group_index_];

        // Try to cast to our custom Options first
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

MCMC::MCMC(gs::SplatData&& splat_data)
    : _splat_data(std::move(splat_data)) {
}

torch::Tensor MCMC::multinomial_sample(const torch::Tensor& weights, int n, bool replacement) {
    const int64_t num_elements = weights.size(0);

    // PyTorch's multinomial has a limit of 2^24 elements
    if (num_elements <= (1 << 24)) {
        return torch::multinomial(weights, n, replacement);
    }
    // For larger arrays, we need to implement sampling manually
    auto weights_normalized = weights / weights.sum();
    auto weights_cpu = weights_normalized.cpu();

    std::vector<int64_t> sampled_indices;
    sampled_indices.reserve(n);

    // Create cumulative distribution
    auto cumsum = weights_cpu.cumsum(0);
    auto cumsum_data = cumsum.accessor<float, 1>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < n; ++i) {
        float u = dis(gen);
        // Binary search for the index
        int64_t idx = 0;
        int64_t left = 0, right = num_elements - 1;
        while (left <= right) {
            int64_t mid = (left + right) / 2;
            if (cumsum_data[mid] < u) {
                left = mid + 1;
            } else {
                idx = mid;
                right = mid - 1;
            }
        }
        sampled_indices.push_back(idx);
    }

    auto result = torch::tensor(sampled_indices, torch::kLong);
    return result.to(weights.device());
}

void MCMC::update_optimizer_for_relocate(torch::optim::Optimizer* optimizer,
                                         const torch::Tensor& sampled_indices,
                                         const torch::Tensor& dead_indices,
                                         int param_position) {
    // Get the parameter
    auto& param = optimizer->param_groups()[param_position].params()[0];
    void* param_key = param.unsafeGetTensorImpl();

    // Check if optimizer state exists
    auto state_it = optimizer->state().find(param_key);
    if (state_it == optimizer->state().end()) {
        // No state exists yet - this can happen if optimizer.step() hasn't been called
        // In this case, there's nothing to reset, so we can safely return
        return;
    }

    // Get the optimizer state - handle both Adam types
    auto& param_state = *state_it->second;
    auto* fused_adam_state = static_cast<gs::FusedAdam::AdamParamState*>(&param_state);
    fused_adam_state->exp_avg.index_put_({sampled_indices}, 0);
    fused_adam_state->exp_avg_sq.index_put_({sampled_indices}, 0);

    if (fused_adam_state->max_exp_avg_sq.defined()) {
        fused_adam_state->max_exp_avg_sq.index_put_({sampled_indices}, 0);
    }
}

int MCMC::relocate_gs() {
    // Get opacities and handle both [N] and [N, 1] shapes
    torch::NoGradGuard no_grad;
    auto opacities = _splat_data.get_opacity();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }

    auto rotation_raw = _splat_data.rotation_raw();
    auto dead_mask = opacities <= _params->min_opacity | (rotation_raw * rotation_raw).sum(-1) < 1e-8f;
    auto dead_indices = dead_mask.nonzero().squeeze(-1);
    int n_dead = dead_indices.numel();

    if (n_dead == 0)
        return 0;

    auto alive_mask = ~dead_mask;
    auto alive_indices = alive_mask.nonzero().squeeze(-1);

    if (alive_indices.numel() == 0)
        return 0;

    // Sample from alive Gaussians based on opacity
    auto probs = opacities.index_select(0, alive_indices);
    auto sampled_idxs_local = multinomial_sample(probs, n_dead, true);
    auto sampled_idxs = alive_indices.index_select(0, sampled_idxs_local);

    // Get parameters for sampled Gaussians
    auto sampled_opacities = opacities.index_select(0, sampled_idxs);
    auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

    // Count occurrences of each sampled index
    auto ratios = torch::ones_like(opacities, torch::kInt32);
    ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kInt32));
    ratios = ratios.index_select(0, sampled_idxs).contiguous();

    // IMPORTANT: Clamp as in Python implementation
    const int n_max = static_cast<int>(_binoms.size(0));
    ratios = torch::clamp_max_(ratios, n_max);

    // Call the CUDA relocation function from gsplat
    auto relocation_result = gsplat::relocation(
        sampled_opacities,
        sampled_scales,
        ratios,
        _binoms,
        n_max);

    auto new_opacities = std::get<0>(relocation_result);
    auto new_scales = std::get<1>(relocation_result);

    // Clamp new opacities
    new_opacities = torch::clamp_(new_opacities, _params->min_opacity, 1.0f - 1e-7f);

    // Update parameters for sampled indices
    // Handle opacity shape properly
    if (_splat_data.opacity_raw().dim() == 2) {
        _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                             torch::logit(new_opacities).unsqueeze(-1));
    } else {
        _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
    }
    _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

    // Copy from sampled to dead indices
    _splat_data.means().index_put_({dead_indices}, _splat_data.means().index_select(0, sampled_idxs));
    _splat_data.sh0().index_put_({dead_indices}, _splat_data.sh0().index_select(0, sampled_idxs));
    _splat_data.shN().index_put_({dead_indices}, _splat_data.shN().index_select(0, sampled_idxs));
    _splat_data.scaling_raw().index_put_({dead_indices}, _splat_data.scaling_raw().index_select(0, sampled_idxs));
    _splat_data.rotation_raw().index_put_({dead_indices}, _splat_data.rotation_raw().index_select(0, sampled_idxs));
    _splat_data.opacity_raw().index_put_({dead_indices}, _splat_data.opacity_raw().index_select(0, sampled_idxs));

    // Update optimizer states for sampled indices
    for (int i = 0; i < 6; ++i) {
        update_optimizer_for_relocate(_optimizer.get(), sampled_idxs, dead_indices, i);
    }

    return n_dead;
}

int MCMC::add_new_gs() {
    // Add this check at the beginning
    torch::NoGradGuard no_grad;
    if (!_optimizer) {
        std::cerr << "Warning: add_new_gs called but optimizer not initialized" << std::endl;
        return 0;
    }

    const int current_n = _splat_data.size();
    const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
    const int n_new = std::max(0, n_target - current_n);

    if (n_new == 0)
        return 0;

    // Get opacities and handle both [N] and [N, 1] shapes
    auto opacities = _splat_data.get_opacity();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }

    auto probs = opacities.flatten();
    auto sampled_idxs = multinomial_sample(probs, n_new, true);

    // Get parameters for sampled Gaussians
    auto sampled_opacities = opacities.index_select(0, sampled_idxs);
    auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

    // Count occurrences
    auto ratios = torch::zeros({opacities.size(0)}, torch::kFloat32).to(torch::kCUDA);
    ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kFloat32));
    ratios = ratios.index_select(0, sampled_idxs) + 1;

    // IMPORTANT: Clamp and convert to int as in Python implementation
    const int n_max = static_cast<int>(_binoms.size(0));
    ratios = torch::clamp(ratios, 1, n_max);
    ratios = ratios.to(torch::kInt32).contiguous(); // Convert to int!

    // Call the CUDA relocation function from gsplat
    auto relocation_result = gsplat::relocation(
        sampled_opacities,
        sampled_scales,
        ratios,
        _binoms,
        n_max);

    auto new_opacities = std::get<0>(relocation_result);
    auto new_scales = std::get<1>(relocation_result);

    // Clamp new opacities
    new_opacities = torch::clamp(new_opacities, _params->min_opacity, 1.0f - 1e-7f);

    // Update existing Gaussians FIRST (before concatenation)
    if (_splat_data.opacity_raw().dim() == 2) {
        _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                             torch::logit(new_opacities).unsqueeze(-1));
    } else {
        _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
    }
    _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

    // Prepare new Gaussians to concatenate
    auto new_means = _splat_data.means().index_select(0, sampled_idxs);
    auto new_sh0 = _splat_data.sh0().index_select(0, sampled_idxs);
    auto new_shN = _splat_data.shN().index_select(0, sampled_idxs);
    auto new_scaling = _splat_data.scaling_raw().index_select(0, sampled_idxs);
    auto new_rotation = _splat_data.rotation_raw().index_select(0, sampled_idxs);
    auto new_opacity = _splat_data.opacity_raw().index_select(0, sampled_idxs);

    // Step 1: Concatenate all parameters
    auto concat_means = torch::cat({_splat_data.means(), new_means}, 0).set_requires_grad(true);
    auto concat_sh0 = torch::cat({_splat_data.sh0(), new_sh0}, 0).set_requires_grad(true);
    auto concat_shN = torch::cat({_splat_data.shN(), new_shN}, 0).set_requires_grad(true);
    auto concat_scaling = torch::cat({_splat_data.scaling_raw(), new_scaling}, 0).set_requires_grad(true);
    auto concat_rotation = torch::cat({_splat_data.rotation_raw(), new_rotation}, 0).set_requires_grad(true);
    auto concat_opacity = torch::cat({_splat_data.opacity_raw(), new_opacity}, 0).set_requires_grad(true);

    // Step 2: SAFER optimizer state update
    // Store the new parameters in a temporary array first
    std::array new_params = {
        &concat_means, &concat_sh0, &concat_shN,
        &concat_scaling, &concat_rotation, &concat_opacity};

    // Collect old parameter keys and states
    std::vector<void*> old_param_keys;
    std::vector<std::unique_ptr<torch::optim::OptimizerParamState>> saved_states;

    for (int i = 0; i < 6; ++i) {
        auto& old_param = _optimizer->param_groups()[i].params()[0];
        void* old_param_key = old_param.unsafeGetTensorImpl();
        old_param_keys.push_back(old_param_key);

        // Check if state exists
        auto state_it = _optimizer->state().find(old_param_key);
        if (state_it == _optimizer->state().end()) {
            saved_states.push_back(nullptr);
        }

        auto* fused_adam_state = static_cast<gs::FusedAdam::AdamParamState*>(state_it->second.get());
        // FusedAdam state
        torch::IntArrayRef new_shape;
        if (i == 0)
            new_shape = new_means.sizes();
        else if (i == 1)
            new_shape = new_sh0.sizes();
        else if (i == 2)
            new_shape = new_shN.sizes();
        else if (i == 3)
            new_shape = new_scaling.sizes();
        else if (i == 4)
            new_shape = new_rotation.sizes();
        else
            new_shape = new_opacity.sizes();

        auto zeros_to_add = torch::zeros(new_shape, fused_adam_state->exp_avg.options());
        auto new_exp_avg = torch::cat({fused_adam_state->exp_avg, zeros_to_add}, 0);
        auto new_exp_avg_sq = torch::cat({fused_adam_state->exp_avg_sq, zeros_to_add}, 0);

        // Create new state
        auto new_state = std::make_unique<gs::FusedAdam::AdamParamState>();
        new_state->step_count = fused_adam_state->step_count;
        new_state->exp_avg = new_exp_avg;
        new_state->exp_avg_sq = new_exp_avg_sq;
        if (fused_adam_state->max_exp_avg_sq.defined()) {
            auto new_max_exp_avg_sq = torch::cat({fused_adam_state->max_exp_avg_sq, zeros_to_add}, 0);
            new_state->max_exp_avg_sq = new_max_exp_avg_sq;
        }

        saved_states.push_back(std::move(new_state));
    }

    // Now remove all old states
    for (auto key : old_param_keys) {
        _optimizer->state().erase(key);
    }

    // Update parameters and add new states
    for (int i = 0; i < 6; ++i) {
        _optimizer->param_groups()[i].params()[0] = *new_params[i];

        if (saved_states[i]) {
            void* new_param_key = new_params[i]->unsafeGetTensorImpl();
            _optimizer->state()[new_param_key] = std::move(saved_states[i]);
        }
    }

    // Step 3: Finally update the model's parameters
    _splat_data.means() = concat_means;
    _splat_data.sh0() = concat_sh0;
    _splat_data.shN() = concat_shN;
    _splat_data.scaling_raw() = concat_scaling;
    _splat_data.rotation_raw() = concat_rotation;
    _splat_data.opacity_raw() = concat_opacity;

    return n_new;
}

void MCMC::inject_noise() {
    torch::NoGradGuard no_grad;

    // Get current learning rate from optimizer (after scheduler has updated it)
    auto& group = _optimizer->param_groups()[0];
    auto* fused_adam_options = static_cast<gs::FusedAdam::Options*>(&group.options());
    const float current_lr = static_cast<float>(fused_adam_options->lr()) * _noise_lr;

    // Generate noise
    auto noise = torch::randn_like(_splat_data.means());

    gsplat::add_noise(
        _splat_data.opacity_raw(),
        _splat_data.scaling_raw(),
        _splat_data.rotation_raw(),
        noise,
        _splat_data.means(),
        current_lr);
}

void MCMC::pre_backward(gs::RenderOutput& render_output) {}

void MCMC::post_backward(int iter, gs::RenderOutput& render_output) {
    // Increment SH degree every 1000 iterations
    torch::NoGradGuard no_grad;
    if (iter % _params->sh_degree_interval == 0) {
        _splat_data.increment_sh_degree();
    }

    // Refine Gaussians
    if (is_refining(iter)) {
        // Relocate dead Gaussians
        relocate_gs();

        // Add new Gaussians
        add_new_gs();
<<<<<<< HEAD

#ifdef _WIN32
        // Windows doesn't support CUDACachingAllocator expandable_segments
        c10::cuda::CUDACachingAllocator::emptyCache();
#endif
=======
>>>>>>> parent of ea456f4 (Fix mcmc memory usage in windows (#274))
    }

    // Inject noise to positions
    inject_noise();
}

void MCMC::step(int iter) {
    if (iter < _params->iterations) {
        auto* fused_adam = dynamic_cast<gs::FusedAdam*>(_optimizer.get());
        fused_adam->step(iter);
        fused_adam->zero_grad(true, iter);
        _scheduler->step();
    }
}

void MCMC::initialize(const gs::param::OptimizationParameters& optimParams) {
    _params = std::make_unique<const gs::param::OptimizationParameters>(optimParams);

    const auto dev = torch::kCUDA;
    _splat_data.means() = _splat_data.means().to(dev).set_requires_grad(true);
    _splat_data.scaling_raw() = _splat_data.scaling_raw().to(dev).set_requires_grad(true);
    _splat_data.rotation_raw() = _splat_data.rotation_raw().to(dev).set_requires_grad(true);
    _splat_data.opacity_raw() = _splat_data.opacity_raw().to(dev).set_requires_grad(true);
    _splat_data.sh0() = _splat_data.sh0().to(dev).set_requires_grad(true);
    _splat_data.shN() = _splat_data.shN().to(dev).set_requires_grad(true);

    // Initialize binomial coefficients
    const int n_max = 51;
    _binoms = torch::zeros({n_max, n_max}, torch::kFloat32);
    auto binoms_accessor = _binoms.accessor<float, 2>();
    for (int n = 0; n < n_max; ++n) {
        for (int k = 0; k <= n; ++k) {
            // Compute binomial coefficient C(n,k)
            float binom = 1.0f;
            for (int i = 0; i < k; ++i) {
                binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
            }
            binoms_accessor[n][k] = binom;
        }
    }
    _binoms = _binoms.to(dev);

    // Initialize optimizer
    using Options = gs::FusedAdam::Options;
    std::vector<torch::optim::OptimizerParamGroup> groups;

    // Create groups with proper unique_ptr<Options>
    auto add_param_group = [&groups](const torch::Tensor& param, double lr) {
        auto options = std::make_unique<Options>(lr);
        options->eps(1e-15).betas(std::make_tuple(0.9, 0.999));
        groups.emplace_back(
            std::vector<torch::Tensor>{param},
            std::unique_ptr<torch::optim::OptimizerOptions>(std::move(options)));
    };

    add_param_group(_splat_data.means(), _params->means_lr * _splat_data.get_scene_scale());
    add_param_group(_splat_data.sh0(), _params->shs_lr);
    add_param_group(_splat_data.shN(), _params->shs_lr / 20.f);
    add_param_group(_splat_data.scaling_raw(), _params->scaling_lr);
    add_param_group(_splat_data.rotation_raw(), _params->rotation_lr);
    add_param_group(_splat_data.opacity_raw(), _params->opacity_lr);

    auto global_options = std::make_unique<Options>(0.f);
    global_options->eps(1e-15);
    _optimizer = std::make_unique<gs::FusedAdam>(std::move(groups), std::move(global_options));
    const double gamma = std::pow(0.01, 1.0 / _params->iterations);
    _scheduler = std::make_unique<ExponentialLR>(*_optimizer, gamma, 0);
}

bool MCMC::is_refining(int iter) const {
    return (iter < _params->stop_refine &&
            iter > _params->start_refine &&
            iter % _params->refine_every == 0);
}
