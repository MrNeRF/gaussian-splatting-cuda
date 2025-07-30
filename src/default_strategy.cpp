#include "core/default_strategy.hpp"
#include "Ops.h"
#include "core/parameters.hpp"
#include "core/rasterizer.hpp"
#include <c10/cuda/CUDACachingAllocator.h>

DefaultStrategy::DefaultStrategy(gs::SplatData&& splat_data)
    : _splat_data(std::move(splat_data)) {
}

void DefaultStrategy::initialize(const gs::param::OptimizationParameters& optimParams) {
    _params = std::make_unique<const gs::param::OptimizationParameters>(optimParams);

    strategy::initialize_gaussians(_splat_data);

    // Initialize optimizer
    _optimizer = strategy::create_optimizer(_splat_data, *_params);

    // Initialize exponential scheduler
    _scheduler = strategy::create_scheduler(*_params, _optimizer.get(), 0);
}

void DefaultStrategy::pre_backward(gs::RenderOutput& render_output) {
    if (_key_for_gradient == "means2d") {
        render_output.means2d.retain_grad();
    }
}

void DefaultStrategy::update_state(gs::RenderOutput& render_output, bool packed) {
    torch::Tensor grads;
    if (_key_for_gradient == "means2d") {
        grads = _absgrad
                    ? render_output.means2d.grad().abs().clone()
                    : render_output.means2d.grad().clone();

        if (!torch::isfinite(grads).all().item<bool>()) {
            throw std::runtime_error("Gradient contains NaN or Inf values.");
        }
    } else {
        throw std::runtime_error("Only means2d is supported for gradient updates in DefaultStrategy.");
    }

    const size_t num_cameras = render_output.image.dim() == 4 ? render_output.image.size(0) : 1;
    const float scale_x = render_output.width / 2.0f * num_cameras;
    const float scale_y = render_output.height / 2.0f * num_cameras;
    grads.select(-1, 0).mul_(scale_x);
    grads.select(-1, 1).mul_(scale_y);

    // Initialize state on the first run
    const size_t num_gaussians = _splat_data.size();
    const c10::Device device = grads.device();
    if (!_grad2d.defined()) {
        _grad2d = torch::zeros(num_gaussians, torch::kFloat32).to(device);
    }
    if (!_count.defined()) {
        _count = torch::zeros(num_gaussians, torch::kFloat32).to(device);
    }
    if (_params->stop_refine_scale2d > 0 && !_radii.defined()) {
        _radii = torch::zeros(num_gaussians, torch::kFloat32).to(device);
    }

    // Update the running state
    torch::Tensor gaussian_ids;
    torch::Tensor radii;
    if (packed) {
        throw std::runtime_error("Packed mode is not supported in this implementation");
        // TODO: Implement packed mode
        // gs_ids = info["gaussian_ids"]  # [nnz]
        // radii = info["radii"].max(dim=-1).values  # [nnz]
    } else {
        // grads is [C, N, 2]
        // Currently, render_output.radii has a shape of [..., N], assuming C = 1
        const torch::Tensor valid_mask = render_output.radii > 0;  // [N]
        gaussian_ids = valid_mask.nonzero().squeeze(-1);           // [nnz]
        grads = grads.squeeze(0).index_select(0, gaussian_ids);    // [nnz, 2]
        radii = render_output.radii.index_select(0, gaussian_ids); // [nnz]
    }

    _grad2d.index_add_(0, gaussian_ids, grads.norm(2, -1));
    _count.index_add_(0, gaussian_ids, torch::ones_like(gaussian_ids, torch::kFloat32));
    if (_params->stop_refine_scale2d > 0) {
        const double max_wh = static_cast<double>(std::max(render_output.width, render_output.height));
        _radii.index_put_({gaussian_ids},
                          torch::max(_radii.index_select(0, gaussian_ids), radii / max_wh));
    }
}

bool DefaultStrategy::is_refining(int iter) const {
    return (iter > _params->start_refine &&
            iter % _params->refine_every == 0 &&
            iter % _params->reset_every >= _params->pause_refine_after_reset);
}

void DefaultStrategy::duplicate(const torch::Tensor is_duplicated) {
    torch::NoGradGuard no_grad;

    const c10::Device device = is_duplicated.device();
    const torch::Tensor sampled_idxs = is_duplicated.nonzero().squeeze(-1);

    const auto param_fn = [&sampled_idxs](const int i, const torch::Tensor param) {
        const torch::Tensor new_param = param.index_select(0, sampled_idxs);
        return torch::cat({param, new_param}).set_requires_grad(param.requires_grad());
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

    // Update the extra running state
    const int num_new_gaussians = sampled_idxs.size(0);
    if (_grad2d.defined()) {
        _grad2d = torch::cat({_grad2d, _grad2d.index_select(0, sampled_idxs)});
    }
    if (_radii.defined()) {
        _radii = torch::cat({_radii, _radii.index_select(0, sampled_idxs)});
    }
    if (_count.defined()) {
        _count = torch::cat({_count, _count.index_select(0, sampled_idxs)});
    }
}

void DefaultStrategy::split(const torch::Tensor is_split) {
    torch::NoGradGuard no_grad;

    const c10::Device device = is_split.device();
    const torch::Tensor sampled_idxs = is_split.nonzero().squeeze(-1);
    const torch::Tensor rest_idxs = is_split.logical_not().nonzero().squeeze(-1);

    const torch::Tensor sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);
    const torch::Tensor sampled_quats = _splat_data.get_rotation().index_select(0, sampled_idxs);
    const torch::Tensor rotmats = gsplat::quats_to_rotmats(sampled_quats); // [N, 3, 3]

    const auto num_split_gaussians = sampled_idxs.size(0);
    const auto split_size = 2;
    const torch::Tensor samples = torch::einsum( // [split_size, N, 3]
        "nij,nj,bnj->bni",
        {rotmats,
         sampled_scales,
         torch::randn({split_size, num_split_gaussians, 3}, sampled_quats.options().device(device))});

    const auto param_fn = [this, &sampled_idxs, &rest_idxs, &samples, &split_size, &sampled_scales](const int i, const torch::Tensor param) {
        std::vector<int64_t> repeats(param.dim(), 1);
        repeats[0] = split_size;

        const torch::Tensor sampled_param = param.index_select(0, sampled_idxs);
        torch::Tensor split_param;
        if (i == 0) {                                                               // means
            split_param = (sampled_param.unsqueeze(0) + samples).reshape({-1, 3});  // [split_size * N, 3]
        } else if (i == 3) {                                                        // scaling
            split_param = torch::log(sampled_scales / 1.6).repeat({split_size, 1}); // [split_size * N, 3]
        } else if (i == 5 && _params->revised_opacity) {                            // opacity
            const torch::Tensor new_opacities = 1.0 - torch::sqrt(1.0 - torch::sigmoid(sampled_param));
            split_param = torch::logit(new_opacities).repeat(repeats); // [split_size * N]
        } else {
            split_param = sampled_param.repeat(repeats);
        }

        const torch::Tensor rest_param = param.index_select(0, rest_idxs);
        return torch::cat({rest_param, split_param}, 0).set_requires_grad(param.requires_grad());
    };

    const auto optimizer_fn = [&sampled_idxs, &rest_idxs, &split_size](
                                  torch::optim::OptimizerParamState& state,
                                  const torch::Tensor full_param)
        -> std::unique_ptr<torch::optim::OptimizerParamState> {
        auto zero_shape = full_param.sizes().vec();
        zero_shape[0] = sampled_idxs.size(0) * split_size;
        if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(&state)) {
            // Standard Adam state
            auto rest_exp_avg = adam_state->exp_avg().index_select(0, rest_idxs);
            auto rest_exp_avg_sq = adam_state->exp_avg_sq().index_select(0, rest_idxs);

            auto zeros_to_add = torch::zeros(zero_shape, adam_state->exp_avg().options());
            auto new_exp_avg = torch::cat({rest_exp_avg, zeros_to_add}, 0);
            auto new_exp_avg_sq = torch::cat({rest_exp_avg_sq, zeros_to_add}, 0);

            // Create new state
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(adam_state->step());
            new_state->exp_avg(new_exp_avg);
            new_state->exp_avg_sq(new_exp_avg_sq);
            if (adam_state->max_exp_avg_sq().defined()) {
                auto rest_max_exp_avg_sq = adam_state->max_exp_avg_sq().index_select(0, rest_idxs);
                auto new_max_exp_avg_sq = torch::cat({rest_max_exp_avg_sq, zeros_to_add}, 0);
                new_state->max_exp_avg_sq(new_max_exp_avg_sq);
            }
            return new_state;
        } else if (auto* selective_adam_state = dynamic_cast<gs::SelectiveAdam::AdamParamState*>(&state)) {
            // SelectiveAdam state
            auto rest_exp_avg = selective_adam_state->exp_avg.index_select(0, rest_idxs);
            auto rest_exp_avg_sq = selective_adam_state->exp_avg_sq.index_select(0, rest_idxs);

            auto zeros_to_add = torch::zeros(zero_shape, selective_adam_state->exp_avg.options());
            auto new_exp_avg = torch::cat({rest_exp_avg, zeros_to_add}, 0);
            auto new_exp_avg_sq = torch::cat({rest_exp_avg_sq, zeros_to_add}, 0);

            // Create new state
            auto new_state = std::make_unique<gs::SelectiveAdam::AdamParamState>();
            new_state->step_count = selective_adam_state->step_count;
            new_state->exp_avg = new_exp_avg;
            new_state->exp_avg_sq = new_exp_avg_sq;
            if (selective_adam_state->max_exp_avg_sq.defined()) {
                auto rest_max_exp_avg_sq = selective_adam_state->max_exp_avg_sq.index_select(0, rest_idxs);
                auto new_max_exp_avg_sq = torch::cat({rest_max_exp_avg_sq, zeros_to_add}, 0);
                new_state->max_exp_avg_sq = new_max_exp_avg_sq;
            }
            return new_state;
        }
        return nullptr;
    };

    strategy::update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);

    // Update the extra running state
    const auto make_repeats = [&split_size](const at::Tensor& t) {
        std::vector<int64_t> v(t.dim(), 1);
        v[0] = split_size;
        return v;
    };
    if (_grad2d.defined()) {
        _grad2d = torch::cat({_grad2d.index_select(0, rest_idxs),
                              _grad2d.index_select(0, sampled_idxs).repeat(make_repeats(_grad2d))});
    }
    if (_radii.defined()) {
        _radii = torch::cat({_radii.index_select(0, rest_idxs),
                             _radii.index_select(0, sampled_idxs).repeat(make_repeats(_radii))});
    }
    if (_count.defined()) {
        _count = torch::cat({_count.index_select(0, rest_idxs),
                             _count.index_select(0, sampled_idxs).repeat(make_repeats(_count))});
    }
}

std::tuple<int64_t, int64_t> DefaultStrategy::grow_gs(int iter) {
    torch::NoGradGuard no_grad;

    const torch::Tensor grads = _grad2d / _count.clamp_min(1);
    const c10::Device device = grads.device();

    const torch::Tensor is_grad_high = grads > _params->grad_threshold;
    const auto max_values = std::get<0>(torch::max(_splat_data.get_scaling(), -1));
    const torch::Tensor is_small = max_values <= _params->grow_scale3d * _splat_data.get_scene_scale();
    const torch::Tensor is_duplicated = is_grad_high & is_small;
    const int64_t num_duplicates = is_duplicated.sum().item<int64_t>();

    const torch::Tensor is_large = ~is_small;
    torch::Tensor is_split = is_grad_high & is_large;
    if (iter < _params->stop_refine_scale2d) {
        is_split |= _radii > _params->grow_scale2d;
    }
    const int64_t num_split = is_split.sum().item<int64_t>();

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

    return {num_duplicates, num_split};
}

void DefaultStrategy::remove(const torch::Tensor is_prune) {
    torch::NoGradGuard no_grad;

    const torch::Tensor sampled_idxs = is_prune.logical_not().nonzero().squeeze(-1);

    const auto param_fn = [&sampled_idxs](const int i, const torch::Tensor param) {
        return param.index_select(0, sampled_idxs).set_requires_grad(param.requires_grad());
    };

    const auto optimizer_fn = [&sampled_idxs](
                                  torch::optim::OptimizerParamState& state,
                                  const torch::Tensor new_param)
        -> std::unique_ptr<torch::optim::OptimizerParamState> {
        if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(&state)) {
            // Standard Adam state
            auto new_exp_avg = adam_state->exp_avg().index_select(0, sampled_idxs);
            auto new_exp_avg_sq = adam_state->exp_avg_sq().index_select(0, sampled_idxs);

            // Create new state
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(adam_state->step());
            new_state->exp_avg(new_exp_avg);
            new_state->exp_avg_sq(new_exp_avg_sq);
            if (adam_state->max_exp_avg_sq().defined()) {
                auto new_max_exp_avg_sq = adam_state->max_exp_avg_sq().index_select(0, sampled_idxs);
                new_state->max_exp_avg_sq(new_max_exp_avg_sq);
            }
            return new_state;
        } else if (auto* selective_adam_state = dynamic_cast<gs::SelectiveAdam::AdamParamState*>(&state)) {
            // SelectiveAdam state
            auto new_exp_avg = selective_adam_state->exp_avg.index_select(0, sampled_idxs);
            auto new_exp_avg_sq = selective_adam_state->exp_avg_sq.index_select(0, sampled_idxs);

            // Create new state
            auto new_state = std::make_unique<gs::SelectiveAdam::AdamParamState>();
            new_state->step_count = selective_adam_state->step_count;
            new_state->exp_avg = new_exp_avg;
            new_state->exp_avg_sq = new_exp_avg_sq;
            if (selective_adam_state->max_exp_avg_sq.defined()) {
                auto new_max_exp_avg_sq = selective_adam_state->max_exp_avg_sq.index_select(0, sampled_idxs);
                new_state->max_exp_avg_sq = new_max_exp_avg_sq;
            }
            return new_state;
        }
        return nullptr;
    };

    strategy::update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data);

    // Update the extra running state
    if (_grad2d.defined()) {
        _grad2d = _grad2d.index_select(0, sampled_idxs);
    }
    if (_radii.defined()) {
        _radii = _radii.index_select(0, sampled_idxs);
    }
    if (_count.defined()) {
        _count = _count.index_select(0, sampled_idxs);
    }
}

int64_t DefaultStrategy::prune_gs(int iter) {
    torch::NoGradGuard no_grad;

    torch::Tensor is_prune = _splat_data.get_opacity() < _params->prune_opacity;
    if (iter > _params->reset_every) {
        const auto max_values = std::get<0>(torch::max(_splat_data.get_scaling(), -1));
        torch::Tensor is_too_big = max_values > _params->prune_scale3d * _splat_data.get_scene_scale();

        if (iter < _params->stop_refine_scale2d) {
            is_too_big |= _radii > _params->prune_scale2d;
        }

        is_prune |= is_too_big;
    }

    const int64_t num_prunes = is_prune.sum().item<int64_t>();
    if (num_prunes > 0) {
        remove(is_prune);
    }
    return num_prunes;
}

void DefaultStrategy::reset_opacity() {
    torch::NoGradGuard no_grad;

    const auto threshold = 2.0f * _params->prune_opacity;

    const auto param_fn = [&threshold](const int i, const torch::Tensor param) {
        if (i == 5) {
            const torch::Tensor new_opacities = torch::clamp_max(
                param,
                torch::logit(torch::tensor(threshold)).item());
            return new_opacities.set_requires_grad(param.requires_grad());
        } else {
            throw std::runtime_error("Invalid parameter index for reset_opacity: " + std::to_string(i));
        }
    };

    const auto optimizer_fn = [](torch::optim::OptimizerParamState& state,
                                 const torch::Tensor new_param)
        -> std::unique_ptr<torch::optim::OptimizerParamState> {
        if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(&state)) {
            // Standard Adam state
            auto new_exp_avg = torch::zeros_like(adam_state->exp_avg());
            auto new_exp_avg_sq = torch::zeros_like(adam_state->exp_avg_sq());

            // Create new state
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(adam_state->step());
            new_state->exp_avg(new_exp_avg);
            new_state->exp_avg_sq(new_exp_avg_sq);
            if (adam_state->max_exp_avg_sq().defined()) {
                auto new_max_exp_avg_sq = torch::zeros_like(adam_state->max_exp_avg_sq());
                new_state->max_exp_avg_sq(new_max_exp_avg_sq);
            }
            return new_state;
        } else if (auto* selective_adam_state = dynamic_cast<gs::SelectiveAdam::AdamParamState*>(&state)) {
            // SelectiveAdam state
            auto new_exp_avg = torch::zeros_like(selective_adam_state->exp_avg);
            auto new_exp_avg_sq = torch::zeros_like(selective_adam_state->exp_avg_sq);

            // Create new state
            auto new_state = std::make_unique<gs::SelectiveAdam::AdamParamState>();
            new_state->step_count = selective_adam_state->step_count;
            new_state->exp_avg = new_exp_avg;
            new_state->exp_avg_sq = new_exp_avg_sq;
            if (selective_adam_state->max_exp_avg_sq.defined()) {
                auto new_max_exp_avg_sq = torch::zeros_like(selective_adam_state->max_exp_avg_sq);
                new_state->max_exp_avg_sq = new_max_exp_avg_sq;
            }
            return new_state;
        }

        return nullptr;
    };

    strategy::update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, _splat_data, {5});
}

void DefaultStrategy::post_backward(int iter, gs::RenderOutput& render_output, bool packed) {
    // Store visibility mask for selective adam
    if (_params->selective_adam) {
        _last_visibility_mask = render_output.visibility;
    }

    // Increment SH degree every 1000 iterations
    torch::NoGradGuard no_grad;
    if (iter % _params->sh_degree_interval == 0) {
        _splat_data.increment_sh_degree();
    }

    if (iter >= _params->stop_refine) {
        return;
    }

    update_state(render_output, packed);

    if (is_refining(iter)) {
        const auto [num_duplicates, num_splits] = grow_gs(iter);
        const auto num_prunes = prune_gs(iter);

        _grad2d.zero_();
        _count.zero_();
        if (_params->stop_refine_scale2d > 0) {
            _radii.zero_();
        }

        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    if (iter % _params->reset_every == 0 && iter > 0) {
        reset_opacity();
    }
}

void DefaultStrategy::step(int iter) {
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
