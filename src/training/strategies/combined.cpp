/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "combined.hpp"
#include "default_strategy.hpp"
#include "mcmc.hpp"
#include "Ops.h"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "optimizers/fused_adam.hpp"
#include "strategy_utils.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include <random>

namespace gs::training {
    using FusedAdam = gs::training::FusedAdam;

    CombinedStrategy::CombinedStrategy(gs::SplatData&& splat_data)
        : _splat_data(std::move(splat_data)) {}

    void CombinedStrategy::ExponentialLR::step() {
        if (param_group_index_ >= 0) {
            auto& group = optimizer_.param_groups()[param_group_index_];
            auto* fused_adam_options = static_cast<FusedAdam::Options*>(&group.options());
            double current_lr = fused_adam_options->lr();
            fused_adam_options->lr(current_lr * gamma_);
        } else {
            for (auto& group : optimizer_.param_groups()) {
                auto* fused_adam_options = static_cast<FusedAdam::Options*>(&group.options());
                double current_lr = fused_adam_options->lr();
                fused_adam_options->lr(current_lr * gamma_);
            }
        }
    }

    void CombinedStrategy::initialize(const gs::param::OptimizationParameters& optimParams) {
        _params = std::make_unique<const gs::param::OptimizationParameters>(optimParams);

        initialize_gaussians(_splat_data);

        using Options = FusedAdam::Options;
        std::vector<torch::optim::OptimizerParamGroup> groups;
        auto add_param_group = [&groups](const torch::Tensor& param, double lr) {
            auto options = std::make_unique<Options>(lr);
            options->eps(1e-15).betas(std::make_tuple(0.9, 0.999));
            groups.emplace_back(std::vector<torch::Tensor>{param},
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
        _optimizer = std::make_unique<FusedAdam>(std::move(groups), std::move(global_options));

        const double gamma = std::pow(0.01, 1.0 / _params->iterations);
        _scheduler = std::make_unique<ExponentialLR>(*_optimizer, gamma, 0);

        // Init MCMC binoms
        const int n_max = 51;
        _binoms = torch::zeros({n_max, n_max}, torch::kFloat32).to(torch::kCUDA);
        auto acc = _binoms.accessor<float, 2>();
        for (int n = 0; n < n_max; ++n) {
            for (int k = 0; k <= n; ++k) {
                float binom = 1.0f;
                for (int i = 0; i < k; ++i) binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
                acc[n][k] = binom;
            }
        }
    }

    bool CombinedStrategy::is_refining(int iter) const {
        return (iter < _params->stop_refine && iter > _params->start_refine && iter % _params->refine_every == 0);
    }

    void CombinedStrategy::remove_gaussians(const torch::Tensor& mask) {
        torch::NoGradGuard no_grad;
        if (mask.sum().item<int>() == 0) return;
        remove(mask);
    }

    void CombinedStrategy::post_backward(int iter, RenderOutput& render_output) {
        torch::NoGradGuard no_grad;
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data.increment_sh_degree();
        }

        // Combine: MCMC-style relocate/add, then Default-style grow/prune, then small noise
        if (is_refining(iter)) {
            relocate_gs();
            add_new_gs();
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
        if (iter % 10 == 0)
            c10::cuda::CUDACachingAllocator::emptyCache();
#endif

        inject_noise();
    }

    void CombinedStrategy::step(int iter) {
        if (iter < _params->iterations) {
            auto* fused_adam = dynamic_cast<FusedAdam*>(_optimizer.get());
            fused_adam->step(iter);
            fused_adam->zero_grad(true, iter);
            _scheduler->step();
        }
    }

    // -------- Default-like helpers (duplicate/split/prune/grow/reset) --------
    void CombinedStrategy::duplicate(const torch::Tensor& is_duplicated) {
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
                auto zeros_to_add = torch::zeros(new_shape, fused_adam_state->exp_avg.options());
                auto new_exp_avg = torch::cat({fused_adam_state->exp_avg, zeros_to_add}, 0);
                auto new_exp_avg_sq = torch::cat({fused_adam_state->exp_avg_sq, zeros_to_add}, 0);
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

    void CombinedStrategy::split(const torch::Tensor& is_split) {
        torch::NoGradGuard no_grad;
        const c10::Device device = is_split.device();
        const torch::Tensor sampled_idxs = is_split.nonzero().squeeze(-1);
        const torch::Tensor rest_idxs = is_split.logical_not().nonzero().squeeze(-1);

        const torch::Tensor sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);
        const torch::Tensor sampled_quats = _splat_data.get_rotation().index_select(0, sampled_idxs);
        const torch::Tensor rotmats = gsplat::quats_to_rotmats(sampled_quats);

        const auto num_split_gaussians = sampled_idxs.size(0);
        constexpr auto split_size = 2;
        const torch::Tensor samples = torch::einsum("nij,nj,bnj->bni",
                                                    {rotmats,
                                                     sampled_scales,
                                                     torch::randn({split_size, num_split_gaussians, 3},
                                                                  sampled_quats.options().device(device))});

        const auto param_fn = [this, &sampled_idxs, &rest_idxs, &samples, &sampled_scales](
                                  const int i, const torch::Tensor& param) {
            std::vector<int64_t> repeats(param.dim(), 1);
            repeats[0] = split_size;
            const torch::Tensor sampled_param = param.index_select(0, sampled_idxs);
            torch::Tensor split_param;
            if (i == 0) {
                split_param = (sampled_param.unsqueeze(0) + samples).reshape({-1, 3});
            } else if (i == 3) {
                split_param = torch::log(sampled_scales / 1.6).repeat({split_size, 1});
            } else if (i == 5 && _params->revised_opacity) {
                const torch::Tensor new_opacities = 1.0 - torch::sqrt(1.0 - torch::sigmoid(sampled_param));
                split_param = torch::logit(new_opacities).repeat(repeats);
            } else {
                split_param = sampled_param.repeat(repeats);
            }
            const torch::Tensor rest_param = param.index_select(0, rest_idxs);
            return torch::cat({rest_param, split_param}, 0).set_requires_grad(param.requires_grad());
        };

        const auto optimizer_fn = [&sampled_idxs, &rest_idxs](torch::optim::OptimizerParamState& state,
                                                              const torch::Tensor& full_param)
            -> std::unique_ptr<torch::optim::OptimizerParamState> {
            auto zero_shape = full_param.sizes().vec();
            zero_shape[0] = sampled_idxs.size(0) * split_size;
            if (auto* fused_adam_state = dynamic_cast<FusedAdam::AdamParamState*>(&state)) {
                auto rest_exp_avg = fused_adam_state->exp_avg.index_select(0, rest_idxs);
                auto rest_exp_avg_sq = fused_adam_state->exp_avg_sq.index_select(0, rest_idxs);
                auto zeros_to_add = torch::zeros(zero_shape, fused_adam_state->exp_avg.options());
                auto new_exp_avg = torch::cat({rest_exp_avg, zeros_to_add}, 0);
                auto new_exp_avg_sq = torch::cat({rest_exp_avg_sq, zeros_to_add}, 0);
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

    void CombinedStrategy::remove(const torch::Tensor& is_prune) {
        torch::NoGradGuard no_grad;
        const torch::Tensor sampled_idxs = is_prune.logical_not().nonzero().squeeze(-1);
        const auto param_fn = [&sampled_idxs](const int i, const torch::Tensor& param) {
            return param.index_select(0, sampled_idxs).set_requires_grad(param.requires_grad());
        };
        const auto optimizer_fn = [&sampled_idxs](torch::optim::OptimizerParamState& state,
                                                  const torch::Tensor& new_param)
            -> std::unique_ptr<torch::optim::OptimizerParamState> {
            if (auto* fused_adam_state = dynamic_cast<FusedAdam::AdamParamState*>(&state)) {
                auto new_exp_avg = fused_adam_state->exp_avg.index_select(0, sampled_idxs);
                auto new_exp_avg_sq = fused_adam_state->exp_avg_sq.index_select(0, sampled_idxs);
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

    void CombinedStrategy::prune_gs(int iter) {
        torch::NoGradGuard no_grad;
        torch::Tensor is_prune = _splat_data.get_opacity() < _params->prune_opacity;
        auto rotation_raw = _splat_data.rotation_raw();
        is_prune |= (rotation_raw * rotation_raw).sum(-1) < 1e-8f;
        if (iter > _params->reset_every) {
            const auto max_values = std::get<0>(torch::max(_splat_data.get_scaling(), -1));
            torch::Tensor is_too_big = max_values > _params->prune_scale3d * _splat_data.get_scene_scale();
            is_prune |= is_too_big;
        }
        const auto num_prunes = is_prune.sum().item<int64_t>();
        if (num_prunes > 0) remove(is_prune);
    }

    void CombinedStrategy::grow_gs(int iter) {
        torch::NoGradGuard no_grad;
        const torch::Tensor grads = _splat_data._densification_info[1] /
                                    torch::clamp_min(_splat_data._densification_info[0], 1.0f);
        const c10::Device device = grads.device();
        const torch::Tensor is_grad_high = grads > _params->grad_threshold;
        const auto max_values = std::get<0>(torch::max(_splat_data.get_scaling(), -1));
        const torch::Tensor is_small = max_values <= _params->grow_scale3d * _splat_data.get_scene_scale();
        const torch::Tensor is_duplicated = is_grad_high & is_small;
        const auto num_duplicates = is_duplicated.sum().item<int64_t>();
        const torch::Tensor is_large = ~is_small;
        torch::Tensor is_split = is_grad_high & is_large;
        const auto num_split = is_split.sum().item<int64_t>();
        if (num_duplicates > 0) duplicate(is_duplicated);
        is_split = torch::cat({is_split,
                               torch::zeros(num_duplicates, c10::TensorOptions().dtype(torch::kBool).device(device))});
        if (num_split > 0) split(is_split);
    }

    void CombinedStrategy::reset_opacity() {
        torch::NoGradGuard no_grad;
        const auto threshold = 2.0f * _params->prune_opacity;
        const auto param_fn = [&threshold](const int i, const torch::Tensor& param) {
            if (i == 5) {
                const torch::Tensor new_opacities = torch::clamp_max(param, torch::logit(torch::tensor(threshold)).item());
                return new_opacities.set_requires_grad(param.requires_grad());
            }
            throw std::runtime_error("Invalid parameter index for reset_opacity: " + std::to_string(i));
        };
        const auto optimizer_fn = [](torch::optim::OptimizerParamState& state, const torch::Tensor& new_param)
            -> std::unique_ptr<torch::optim::OptimizerParamState> {
            if (auto* fused_adam_state = dynamic_cast<FusedAdam::AdamParamState*>(&state)) {
                auto new_exp_avg = torch::zeros_like(fused_adam_state->exp_avg);
                auto new_exp_avg_sq = torch::zeros_like(fused_adam_state->exp_avg_sq);
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

    // -------- MCMC-like helpers (relocate/add/noise) --------
    torch::Tensor CombinedStrategy::multinomial_sample(const torch::Tensor& weights, int n, bool replacement) {
        const int64_t num_elements = weights.size(0);
        if (num_elements <= (1 << 24)) return torch::multinomial(weights, n, replacement);
        auto weights_normalized = weights / weights.sum();
        auto weights_cpu = weights_normalized.cpu();
        std::vector<int64_t> sampled_indices; sampled_indices.reserve(n);
        auto cumsum = weights_cpu.cumsum(0); auto cumsum_data = cumsum.accessor<float, 1>();
        std::random_device rd; std::mt19937 gen(rd()); std::uniform_real_distribution<float> dis(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            float u = dis(gen); int64_t idx = 0; int64_t left = 0, right = num_elements - 1;
            while (left <= right) { int64_t mid = (left + right) / 2; if (cumsum_data[mid] < u) left = mid + 1; else { idx = mid; right = mid - 1; } }
            sampled_indices.push_back(idx);
        }
        auto result = torch::tensor(sampled_indices, torch::kLong);
        return result.to(weights.device());
    }

    void CombinedStrategy::update_optimizer_for_relocate(torch::optim::Optimizer* optimizer,
                                                         const torch::Tensor& sampled_indices,
                                                         const torch::Tensor& dead_indices,
                                                         int param_position) {
        auto& param = optimizer->param_groups()[param_position].params()[0];
        void* param_key = param.unsafeGetTensorImpl();
        auto state_it = optimizer->state().find(param_key);
        if (state_it == optimizer->state().end()) return;
        auto& param_state = *state_it->second;
        auto* fused_adam_state = static_cast<FusedAdam::AdamParamState*>(&param_state);
        fused_adam_state->exp_avg.index_put_({sampled_indices}, 0);
        fused_adam_state->exp_avg_sq.index_put_({sampled_indices}, 0);
        if (fused_adam_state->max_exp_avg_sq.defined()) fused_adam_state->max_exp_avg_sq.index_put_({sampled_indices}, 0);
    }

    int CombinedStrategy::relocate_gs() {
        torch::NoGradGuard no_grad;
        auto opacities = _splat_data.get_opacity();
        if (opacities.dim() == 2 && opacities.size(1) == 1) opacities = opacities.squeeze(-1);
        auto rotation_raw = _splat_data.rotation_raw();
        auto dead_mask = opacities <= _params->min_opacity | (rotation_raw * rotation_raw).sum(-1) < 1e-8f;
        auto dead_indices = dead_mask.nonzero().squeeze(-1);
        int n_dead = dead_indices.numel(); if (n_dead == 0) return 0;
        auto alive_mask = ~dead_mask; auto alive_indices = alive_mask.nonzero().squeeze(-1);
        if (alive_indices.numel() == 0) return 0;
        auto probs = opacities.index_select(0, alive_indices);
        auto sampled_idxs_local = multinomial_sample(probs, n_dead, true);
        auto sampled_idxs = alive_indices.index_select(0, sampled_idxs_local);
        auto sampled_opacities = opacities.index_select(0, sampled_idxs);
        auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);
        auto ratios = torch::ones_like(opacities, torch::kInt32);
        ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kInt32));
        ratios = ratios.index_select(0, sampled_idxs).contiguous();
        const int n_max = static_cast<int>(_binoms.size(0));
        ratios = torch::clamp_max_(ratios, n_max);
        auto relocation_result = gsplat::relocation(sampled_opacities, sampled_scales, ratios, _binoms, n_max);
        auto new_opacities = std::get<0>(relocation_result);
        auto new_scales = std::get<1>(relocation_result);
        new_opacities = torch::clamp_(new_opacities, _params->min_opacity, 1.0f - 1e-7f);
        if (_splat_data.opacity_raw().dim() == 2) {
            _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()}, torch::logit(new_opacities).unsqueeze(-1));
        } else {
            _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
        }
        _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));
        _splat_data.means().index_put_({dead_indices}, _splat_data.means().index_select(0, sampled_idxs));
        _splat_data.sh0().index_put_({dead_indices}, _splat_data.sh0().index_select(0, sampled_idxs));
        _splat_data.shN().index_put_({dead_indices}, _splat_data.shN().index_select(0, sampled_idxs));
        _splat_data.scaling_raw().index_put_({dead_indices}, _splat_data.scaling_raw().index_select(0, sampled_idxs));
        _splat_data.rotation_raw().index_put_({dead_indices}, _splat_data.rotation_raw().index_select(0, sampled_idxs));
        _splat_data.opacity_raw().index_put_({dead_indices}, _splat_data.opacity_raw().index_select(0, sampled_idxs));
        for (int i = 0; i < 6; ++i) update_optimizer_for_relocate(_optimizer.get(), sampled_idxs, dead_indices, i);
        return n_dead;
    }

    int CombinedStrategy::add_new_gs() {
        torch::NoGradGuard no_grad;
        if (!_optimizer) return 0;
        const int current_n = _splat_data.size();
        const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
        const int n_new = std::max(0, n_target - current_n);
        if (n_new == 0) return 0;
        auto opacities = _splat_data.get_opacity();
        if (opacities.dim() == 2 && opacities.size(1) == 1) opacities = opacities.squeeze(-1);
        auto probs = opacities.flatten();
        auto sampled_idxs = multinomial_sample(probs, n_new, true);
        auto sampled_opacities = opacities.index_select(0, sampled_idxs);
        auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);
        auto ratios = torch::zeros({opacities.size(0)}, torch::kFloat32).to(torch::kCUDA);
        ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kFloat32));
        ratios = ratios.index_select(0, sampled_idxs) + 1;
        const int n_max = static_cast<int>(_binoms.size(0));
        ratios = torch::clamp(ratios, 1, n_max).to(torch::kInt32).contiguous();
        auto relocation_result = gsplat::relocation(sampled_opacities, sampled_scales, ratios, _binoms, n_max);
        auto new_opacities = std::get<0>(relocation_result);
        auto new_scales = std::get<1>(relocation_result);
        new_opacities = torch::clamp(new_opacities, _params->min_opacity, 1.0f - 1e-7f);
        if (_splat_data.opacity_raw().dim() == 2) {
            _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()}, torch::logit(new_opacities).unsqueeze(-1));
        } else {
            _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
        }
        _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));
        auto new_means = _splat_data.means().index_select(0, sampled_idxs);
        auto new_sh0 = _splat_data.sh0().index_select(0, sampled_idxs);
        auto new_shN = _splat_data.shN().index_select(0, sampled_idxs);
        auto new_scaling = _splat_data.scaling_raw().index_select(0, sampled_idxs);
        auto new_rotation = _splat_data.rotation_raw().index_select(0, sampled_idxs);
        auto new_opacity = _splat_data.opacity_raw().index_select(0, sampled_idxs);
        auto concat_means = torch::cat({_splat_data.means(), new_means}, 0).set_requires_grad(true);
        auto concat_sh0 = torch::cat({_splat_data.sh0(), new_sh0}, 0).set_requires_grad(true);
        auto concat_shN = torch::cat({_splat_data.shN(), new_shN}, 0).set_requires_grad(true);
        auto concat_scaling = torch::cat({_splat_data.scaling_raw(), new_scaling}, 0).set_requires_grad(true);
        auto concat_rotation = torch::cat({_splat_data.rotation_raw(), new_rotation}, 0).set_requires_grad(true);
        auto concat_opacity = torch::cat({_splat_data.opacity_raw(), new_opacity}, 0).set_requires_grad(true);
        std::array new_params = {&concat_means, &concat_sh0, &concat_shN, &concat_scaling, &concat_rotation, &concat_opacity};
        std::vector<void*> old_param_keys; std::vector<std::unique_ptr<torch::optim::OptimizerParamState>> saved_states;
        for (int i = 0; i < 6; ++i) {
            auto& old_param = _optimizer->param_groups()[i].params()[0];
            void* old_param_key = old_param.unsafeGetTensorImpl();
            old_param_keys.push_back(old_param_key);
            auto state_it = _optimizer->state().find(old_param_key);
            if (state_it == _optimizer->state().end()) { saved_states.push_back(nullptr); continue; }
            auto* fused_adam_state = static_cast<FusedAdam::AdamParamState*>(state_it->second.get());
            torch::IntArrayRef new_shape;
            if (i == 0) new_shape = new_means.sizes();
            else if (i == 1) new_shape = new_sh0.sizes();
            else if (i == 2) new_shape = new_shN.sizes();
            else if (i == 3) new_shape = new_scaling.sizes();
            else if (i == 4) new_shape = new_rotation.sizes();
            else new_shape = new_opacity.sizes();
            auto zeros_to_add = torch::zeros(new_shape, fused_adam_state->exp_avg.options());
            auto new_exp_avg = torch::cat({fused_adam_state->exp_avg, zeros_to_add}, 0);
            auto new_exp_avg_sq = torch::cat({fused_adam_state->exp_avg_sq, zeros_to_add}, 0);
            auto new_state = std::make_unique<FusedAdam::AdamParamState>();
            new_state->step_count = fused_adam_state->step_count;
            new_state->exp_avg = new_exp_avg;
            new_state->exp_avg_sq = new_exp_avg_sq;
            if (fused_adam_state->max_exp_avg_sq.defined()) {
                auto new_max_exp_avg_sq = torch::cat({fused_adam_state->max_exp_avg_sq, zeros_to_add}, 0);
                new_state->max_exp_avg_sq = new_max_exp_avg_sq;
            }
            saved_states.push_back(std::move(new_state));
        }
        for (auto key : old_param_keys) _optimizer->state().erase(key);
        for (int i = 0; i < 6; ++i) {
            _optimizer->param_groups()[i].params()[0] = *new_params[i];
            if (saved_states[i]) {
                void* new_param_key = new_params[i]->unsafeGetTensorImpl();
                _optimizer->state()[new_param_key] = std::move(saved_states[i]);
            }
        }
        _splat_data.means() = concat_means; _splat_data.sh0() = concat_sh0; _splat_data.shN() = concat_shN;
        _splat_data.scaling_raw() = concat_scaling; _splat_data.rotation_raw() = concat_rotation; _splat_data.opacity_raw() = concat_opacity;
        return n_new;
    }

    void CombinedStrategy::inject_noise() {
        torch::NoGradGuard no_grad;
        auto& group = _optimizer->param_groups()[0];
        auto* fused_adam_options = static_cast<FusedAdam::Options*>(&group.options());
        const float current_lr = static_cast<float>(fused_adam_options->lr()) * _noise_lr;
        auto noise = torch::randn_like(_splat_data.means());
        gsplat::add_noise(_splat_data.opacity_raw(), _splat_data.scaling_raw(), _splat_data.rotation_raw(), noise, _splat_data.means(), current_lr);
    }
} // namespace gs::training


