#include "core/mcmc.hpp"
#include "Ops.h"
#include "core/debug_utils.hpp"
#include "core/parameters.hpp"
#include "core/rasterizer.hpp"
#include "core/selective_adam.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include <exception>
#include <iostream>
#include <random>

MCMC::MCMC(gs::SplatData&& splat_data)
    : _splat_data(std::move(splat_data)) {
}

torch::Tensor MCMC::multinomial_sample(const torch::Tensor& weights, int n, bool replacement) {
    const int64_t num_elements = weights.size(0);

    // PyTorch's multinomial has a limit of 2^24 elements
    if (num_elements <= (1 << 24)) {
        return torch::multinomial(weights, n, replacement);
    } else {
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

    if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(&param_state)) {
        // Standard Adam
        adam_state->exp_avg().index_put_({sampled_indices}, 0);
        adam_state->exp_avg_sq().index_put_({sampled_indices}, 0);

        if (adam_state->max_exp_avg_sq().defined()) {
            adam_state->max_exp_avg_sq().index_put_({sampled_indices}, 0);
        }
    } else if (auto* selective_adam_state = dynamic_cast<gs::SelectiveAdam::AdamParamState*>(&param_state)) {
        // SelectiveAdam
        selective_adam_state->exp_avg.index_put_({sampled_indices}, 0);
        selective_adam_state->exp_avg_sq.index_put_({sampled_indices}, 0);

        if (selective_adam_state->max_exp_avg_sq.defined()) {
            selective_adam_state->max_exp_avg_sq.index_put_({sampled_indices}, 0);
        }
    }
}

int MCMC::relocate_gs() {
    // Get opacities and handle both [N] and [N, 1] shapes
    torch::NoGradGuard no_grad;
    auto opacities = _splat_data.get_opacity();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }

    auto dead_mask = opacities <= _params->min_opacity;
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

    const auto param_fn = [&sampled_idxs](const int i, const torch::Tensor param) {
        const torch::Tensor new_param = param.index_select(0, sampled_idxs);
        return torch::cat({param, new_param}, 0).set_requires_grad(param.requires_grad());
    };

    const auto optimizer_fn = [&sampled_idxs](torch::optim::OptimizerParamState& state,
                                              const torch::Tensor full_param)
        -> std::unique_ptr<torch::optim::OptimizerParamState> {
        auto new_shape = full_param.sizes().vec();
        new_shape[0] = sampled_idxs.size(0);
        if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(&state)) {
            // Standard Adam state
            auto zeros_to_add = torch::zeros(new_shape, adam_state->exp_avg().options());
            auto new_exp_avg = torch::cat({adam_state->exp_avg(), zeros_to_add}, 0);
            auto new_exp_avg_sq = torch::cat({adam_state->exp_avg_sq(), zeros_to_add}, 0);

            // Create new state
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(adam_state->step());
            new_state->exp_avg(new_exp_avg);
            new_state->exp_avg_sq(new_exp_avg_sq);
            if (adam_state->max_exp_avg_sq().defined()) {
                auto new_max_exp_avg_sq = torch::cat({adam_state->max_exp_avg_sq(), zeros_to_add}, 0);
                new_state->max_exp_avg_sq(new_max_exp_avg_sq);
            }
            return new_state;
        } else if (auto* selective_adam_state = dynamic_cast<gs::SelectiveAdam::AdamParamState*>(&state)) {
            // SelectiveAdam state
            auto zeros_to_add = torch::zeros(new_shape, selective_adam_state->exp_avg.options());
            auto new_exp_avg = torch::cat({selective_adam_state->exp_avg, zeros_to_add}, 0);
            auto new_exp_avg_sq = torch::cat({selective_adam_state->exp_avg_sq, zeros_to_add}, 0);

            // Create new state
            auto new_state = std::make_unique<gs::SelectiveAdam::AdamParamState>();
            new_state->step_count = selective_adam_state->step_count;
            new_state->exp_avg = new_exp_avg;
            new_state->exp_avg_sq = new_exp_avg_sq;
            if (selective_adam_state->max_exp_avg_sq.defined()) {
                auto new_max_exp_avg_sq = torch::cat({selective_adam_state->max_exp_avg_sq, zeros_to_add}, 0);
                new_state->max_exp_avg_sq = new_max_exp_avg_sq;
            }
            return new_state;
        }
        return nullptr;
    };

    strategy::update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);

    return n_new;
}

void MCMC::inject_noise() {
    // Get opacities and handle both [N] and [N, 1] shapes
    torch::NoGradGuard no_grad;

    auto opacities = _splat_data.get_opacity();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }

    auto scales = _splat_data.get_scaling();
    auto quats = _splat_data.get_rotation();

    // Use gsplat's quat_scale_to_covar_preci function
    auto covar_result = gsplat::quat_scale_to_covar_preci_fwd(
        quats,
        scales,
        true,  // compute_covar
        false, // compute_preci
        false  // triu
    );
    auto covars = std::get<0>(covar_result); // [N, 3, 3]

    // Opacity sigmoid function: 1 / (1 + exp(-k * (x - x0)))
    const float k = 100.0f;
    const float x0 = 0.995f;
    auto op_sigmoid = 1.0f / (1.0f + torch::exp(-k * ((1.0f - opacities) - x0)));

    // Get current learning rate from optimizer (after scheduler has updated it)
    float current_lr = 0.0f;
    auto& group = _optimizer->param_groups()[0];
    if (auto* adam_options = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
        current_lr = static_cast<float>(adam_options->lr());
    } else if (auto* selective_adam_options = dynamic_cast<gs::SelectiveAdam::Options*>(&group.options())) {
        current_lr = static_cast<float>(selective_adam_options->lr());
    }

    // Generate noise
    auto noise = torch::randn_like(_splat_data.means()) * op_sigmoid.unsqueeze(-1) * current_lr * _noise_lr;

    // Transform noise by covariance
    noise = torch::bmm(covars, noise.unsqueeze(-1)).squeeze(-1);

    // Add noise to positions
    _splat_data.means().add_(noise);
}

void MCMC::pre_backward(gs::RenderOutput& render_output) {}

void MCMC::post_backward(int iter, gs::RenderOutput& render_output, bool packed) {
    // Store visibility mask for selective adam
    if (_params->selective_adam) {
        _last_visibility_mask = render_output.visibility;
    }

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

        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    // Inject noise to positions
    inject_noise();
}

void MCMC::step(int iter) {
    if (iter < _params->iterations) {
        if (_params->selective_adam && _last_visibility_mask.defined()) {
            auto* selective_adam = dynamic_cast<gs::SelectiveAdam*>(_optimizer.get());
            if (selective_adam) {
                selective_adam->step(_last_visibility_mask);
            } else {
                _optimizer->step();
            }
        } else {
            _optimizer->step();
        }
        _optimizer->zero_grad(true);
        _scheduler->step();
    }
}

void MCMC::initialize(const gs::param::OptimizationParameters& optimParams) {
    _params = std::make_unique<const gs::param::OptimizationParameters>(optimParams);

    strategy::initialize_gaussians(_splat_data);

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
    _binoms = _binoms.to(torch::kCUDA);

    // Initialize optimizer
    _optimizer = strategy::create_optimizer(_splat_data, *_params);

    // Initialize exponential scheduler
    _scheduler = strategy::create_scheduler(*_params, _optimizer.get(), 0);
}

bool MCMC::is_refining(int iter) const {
    return (iter < _params->stop_refine &&
            iter > _params->start_refine &&
            iter % _params->refine_every == 0);
}
