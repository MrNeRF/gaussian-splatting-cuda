/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "default_strategy.hpp"
#include "Ops.h"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "optimizers/fused_adam.hpp"
#include "rasterization/rasterizer.hpp"
#include "strategy_utils.hpp"
#include <c10/cuda/CUDACachingAllocator.h>

namespace gs::training {
    DefaultStrategy::DefaultStrategy(gs::SplatData&& splat_data)
        : _splat_data(std::move(splat_data)) {
    }

    void DefaultStrategy::initialize(const gs::param::OptimizationParameters& optimParams) {
        _params = std::make_unique<const gs::param::OptimizationParameters>(optimParams);

        initialize_gaussians(_splat_data);

        // Initialize optimizer
        _optimizer = create_optimizer(_splat_data, *_params);

        // Initialize exponential scheduler
        _scheduler = create_scheduler(*_params, _optimizer.get(), 0);
    }

    bool DefaultStrategy::is_refining(int iter) const {
        return (iter > _params->start_refine &&
                iter % _params->refine_every == 0 &&
                iter % _params->reset_every >= _params->pause_refine_after_reset);
    }

    void DefaultStrategy::remove_gaussians(const torch::Tensor& mask) {
        torch::NoGradGuard no_grad;

        if (mask.sum().item<int>() == 0) {
            LOG_DEBUG("No Gaussians to remove");
            return;
        }

        LOG_DEBUG("Removing {} Gaussians", mask.sum().item<int>());
        remove(mask);
    }

    void DefaultStrategy::duplicate(const torch::Tensor& is_duplicated) {
        torch::NoGradGuard no_grad;

        const torch::Tensor sampled_idxs = is_duplicated.nonzero().squeeze(-1);

        const auto param_fn = [&sampled_idxs](const int i, const torch::Tensor& param) {
            const torch::Tensor new_param = param.index_select(0, sampled_idxs);
            return torch::cat({param, new_param}).set_requires_grad(param.requires_grad());
        };

        const auto optimizer_fn = [&sampled_idxs](torch::optim::OptimizerParamState& state,
                                                  const torch::Tensor& full_param)
            -> std::unique_ptr<torch::optim::OptimizerParamState> {
            auto new_shape = full_param.sizes().vec();
            new_shape[0] = sampled_idxs.size(0);
            if (auto* fused_adam_state = dynamic_cast<FusedAdam::AdamParamState*>(&state)) {
                // FusedAdam state
                auto zeros_to_add = torch::zeros(new_shape, fused_adam_state->exp_avg.options());
                auto new_exp_avg = torch::cat({fused_adam_state->exp_avg, zeros_to_add}, 0);
                auto new_exp_avg_sq = torch::cat({fused_adam_state->exp_avg_sq, zeros_to_add}, 0);

                // Create new state
                auto new_state = std::make_unique<FusedAdam::AdamParamState>();
                new_state->step_count = fused_adam_state->step_count;
                new_state->exp_avg = new_exp_avg;
                new_state->exp_avg_sq = new_exp_avg_sq;
                if (fused_adam_state->max_exp_avg_sq.defined()) {
                    auto new_max_exp_avg_sq = torch::cat({fused_adam_state->max_exp_avg_sq, zeros_to_add}, 0);
                    new_state->max_exp_avg_sq = new_max_exp_avg_sq;
                }
                return new_state;
            }
            return nullptr;
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);
    }

    void DefaultStrategy::split(const torch::Tensor& is_split) {
        torch::NoGradGuard no_grad;

        const c10::Device device = is_split.device();
        const torch::Tensor sampled_idxs = is_split.nonzero().squeeze(-1);
        const torch::Tensor rest_idxs = is_split.logical_not().nonzero().squeeze(-1);

        const torch::Tensor sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);
        const torch::Tensor sampled_quats = _splat_data.get_rotation().index_select(0, sampled_idxs);
        const torch::Tensor rotmats = gsplat::quats_to_rotmats(sampled_quats); // [N, 3, 3]

        const auto num_split_gaussians = sampled_idxs.size(0);
        constexpr auto split_size = 2;
        const torch::Tensor samples = torch::einsum( // [split_size, N, 3]
            "nij,nj,bnj->bni",
            {rotmats,
             sampled_scales,
             torch::randn({split_size, num_split_gaussians, 3}, sampled_quats.options().device(device))});

        const auto param_fn = [this, &sampled_idxs, &rest_idxs, &samples, &sampled_scales](
                                  const int i, const torch::Tensor& param) {
            std::vector<int64_t> repeats(param.dim(), 1);
            repeats[0] = split_size;

            const torch::Tensor sampled_param = param.index_select(0, sampled_idxs);
            torch::Tensor split_param;
            if (i == 0) {
                // means
                split_param = (sampled_param.unsqueeze(0) + samples).reshape({-1, 3}); // [split_size * N, 3]
            } else if (i == 3) {
                // scaling
                split_param = torch::log(sampled_scales / 1.6).repeat({split_size, 1}); // [split_size * N, 3]
            } else if (i == 5 && _params->revised_opacity) {
                // opacity
                const torch::Tensor new_opacities = 1.0 - torch::sqrt(1.0 - torch::sigmoid(sampled_param));
                split_param = torch::logit(new_opacities).repeat(repeats); // [split_size * N]
            } else {
                split_param = sampled_param.repeat(repeats);
            }

            const torch::Tensor rest_param = param.index_select(0, rest_idxs);
            return torch::cat({rest_param, split_param}, 0).set_requires_grad(param.requires_grad());
        };

        const auto optimizer_fn = [&sampled_idxs, &rest_idxs](
                                      torch::optim::OptimizerParamState& state,
                                      const torch::Tensor& full_param)
            -> std::unique_ptr<torch::optim::OptimizerParamState> {
            auto zero_shape = full_param.sizes().vec();
            zero_shape[0] = sampled_idxs.size(0) * split_size;
            if (auto* fused_adam_state = dynamic_cast<FusedAdam::AdamParamState*>(&state)) {
                // FusedAdam state
                auto rest_exp_avg = fused_adam_state->exp_avg.index_select(0, rest_idxs);
                auto rest_exp_avg_sq = fused_adam_state->exp_avg_sq.index_select(0, rest_idxs);

                auto zeros_to_add = torch::zeros(zero_shape, fused_adam_state->exp_avg.options());
                auto new_exp_avg = torch::cat({rest_exp_avg, zeros_to_add}, 0);
                auto new_exp_avg_sq = torch::cat({rest_exp_avg_sq, zeros_to_add}, 0);

                // Create new state
                auto new_state = std::make_unique<FusedAdam::AdamParamState>();
                new_state->step_count = fused_adam_state->step_count;
                new_state->exp_avg = new_exp_avg;
                new_state->exp_avg_sq = new_exp_avg_sq;
                if (fused_adam_state->max_exp_avg_sq.defined()) {
                    auto rest_max_exp_avg_sq = fused_adam_state->max_exp_avg_sq.index_select(0, rest_idxs);
                    auto new_max_exp_avg_sq = torch::cat({rest_max_exp_avg_sq, zeros_to_add}, 0);
                    new_state->max_exp_avg_sq = new_max_exp_avg_sq;
                }
                return new_state;
            }
            return nullptr;
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);
    }

    void DefaultStrategy::grow_gs(int iter) {
        torch::NoGradGuard no_grad;

        const torch::Tensor grads = _splat_data._densification_info[1] / torch::clamp_min(
                                                                             _splat_data._densification_info[0], 1.0f);
        const c10::Device device = grads.device();

        const torch::Tensor is_grad_high = grads > _params->grad_threshold;
        const auto max_values = std::get<0>(torch::max(_splat_data.get_scaling(), -1));
        const torch::Tensor is_small = max_values <= _params->grow_scale3d * _splat_data.get_scene_scale();
        const torch::Tensor is_duplicated = is_grad_high & is_small;
        const auto num_duplicates = is_duplicated.sum().item<int64_t>();

        const torch::Tensor is_large = ~is_small;
        torch::Tensor is_split = is_grad_high & is_large;
        const auto num_split = is_split.sum().item<int64_t>();

        // First duplicate
        if (num_duplicates > 0) {
            duplicate(is_duplicated);
        }

        // New Gaussians added by duplication will not be split
        is_split = torch::cat({is_split,
                               torch::zeros(num_duplicates, c10::TensorOptions().dtype(torch::kBool).device(device))});
        if (num_split > 0) {
            split(is_split);
        }
    }

    void DefaultStrategy::remove(const torch::Tensor& is_prune) {
        torch::NoGradGuard no_grad;

        const torch::Tensor sampled_idxs = is_prune.logical_not().nonzero().squeeze(-1);

        const auto param_fn = [&sampled_idxs](const int i, const torch::Tensor& param) {
            return param.index_select(0, sampled_idxs).set_requires_grad(param.requires_grad());
        };

        const auto optimizer_fn = [&sampled_idxs](
                                      torch::optim::OptimizerParamState& state,
                                      const torch::Tensor& new_param)
            -> std::unique_ptr<torch::optim::OptimizerParamState> {
            if (auto* fused_adam_state = dynamic_cast<FusedAdam::AdamParamState*>(&state)) {
                // FusedAdam state
                auto new_exp_avg = fused_adam_state->exp_avg.index_select(0, sampled_idxs);
                auto new_exp_avg_sq = fused_adam_state->exp_avg_sq.index_select(0, sampled_idxs);

                // Create new state
                auto new_state = std::make_unique<FusedAdam::AdamParamState>();
                new_state->step_count = fused_adam_state->step_count;
                new_state->exp_avg = new_exp_avg;
                new_state->exp_avg_sq = new_exp_avg_sq;
                if (fused_adam_state->max_exp_avg_sq.defined()) {
                    auto new_max_exp_avg_sq = fused_adam_state->max_exp_avg_sq.index_select(0, sampled_idxs);
                    new_state->max_exp_avg_sq = new_max_exp_avg_sq;
                }
                return new_state;
            }
            return nullptr;
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);
    }

    void DefaultStrategy::prune_gs(int iter) {
        torch::NoGradGuard no_grad;

        // Check for low opacity
        torch::Tensor is_prune = _splat_data.get_opacity() < _params->prune_opacity;

        auto rotation_raw = _splat_data.rotation_raw();
        is_prune |= (rotation_raw * rotation_raw).sum(-1) < 1e-8f;

        // Check for too large Gaussians
        if (iter > _params->reset_every) {
            const auto max_values = std::get<0>(torch::max(_splat_data.get_scaling(), -1));
            torch::Tensor is_too_big = max_values > _params->prune_scale3d * _splat_data.get_scene_scale();
            is_prune |= is_too_big;
        }

        const auto num_prunes = is_prune.sum().item<int64_t>();
        if (num_prunes > 0) {
            remove(is_prune);
        }
    }

    void DefaultStrategy::reset_opacity() {
        torch::NoGradGuard no_grad;

        const auto threshold = 2.0f * _params->prune_opacity;

        const auto param_fn = [&threshold](const int i, const torch::Tensor& param) {
            if (i == 5) {
                const torch::Tensor new_opacities = torch::clamp_max(
                    param,
                    torch::logit(torch::tensor(threshold)).item());
                return new_opacities.set_requires_grad(param.requires_grad());
            }
            throw std::runtime_error("Invalid parameter index for reset_opacity: " + std::to_string(i));
        };

        const auto optimizer_fn = [](torch::optim::OptimizerParamState& state,
                                     const torch::Tensor& new_param)
            -> std::unique_ptr<torch::optim::OptimizerParamState> {
            if (auto* fused_adam_state = dynamic_cast<FusedAdam::AdamParamState*>(&state)) {
                // FusedAdam state
                auto new_exp_avg = torch::zeros_like(fused_adam_state->exp_avg);
                auto new_exp_avg_sq = torch::zeros_like(fused_adam_state->exp_avg_sq);

                // Create new state
                auto new_state = std::make_unique<FusedAdam::AdamParamState>();
                new_state->step_count = fused_adam_state->step_count;
                new_state->exp_avg = new_exp_avg;
                new_state->exp_avg_sq = new_exp_avg_sq;
                if (fused_adam_state->max_exp_avg_sq.defined()) {
                    auto new_max_exp_avg_sq = torch::zeros_like(fused_adam_state->max_exp_avg_sq);
                    new_state->max_exp_avg_sq = new_max_exp_avg_sq;
                }
                return new_state;
            }

            return nullptr;
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data, {5});
    }

    void DefaultStrategy::post_backward(int iter, RenderOutput& render_output) {
        // Increment SH degree every 1000 iterations
        torch::NoGradGuard no_grad;
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data.increment_sh_degree();
        }

        if (iter == _params->stop_refine) {
            // Reset densification info at the end of refinement.Saves memory and processing time.
            _splat_data._densification_info = torch::empty({0});
        }

        if (iter >= _params->stop_refine) {
            return;
        }

        if (is_refining(iter)) {
            grow_gs(iter);
            prune_gs(iter);

            _splat_data._densification_info = torch::zeros({2, _splat_data.means().size(0)},
                                                           _splat_data.means().options())
                                                  .set_requires_grad(false);
        }

        if (iter % _params->reset_every == 0 && iter > 0) {
            reset_opacity();
        }

#ifdef _WIN32
        // Windows doesn't support CUDACachingAllocator expandable_segments
        if (iter % 10 == 0)
            c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    }

    void DefaultStrategy::step(int iter) {
        if (iter < _params->iterations) {
            auto* fused_adam = dynamic_cast<FusedAdam*>(_optimizer.get());
            fused_adam->step(iter);
            fused_adam->zero_grad(true, iter);
            _scheduler->step();
        }
    }
} // namespace gs::training