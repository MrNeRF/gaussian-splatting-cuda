#include "core/mcmc.hpp"
#include "Ops.h"
#include "core/debug_utils.hpp"
#include "core/parameters.hpp"
#include "core/rasterizer.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include <exception>
#include <iostream>
#include <random>

// Helper function to build rotation matrix from quaternion
static inline torch::Tensor build_rotation(torch::Tensor r) {
    torch::Tensor norm = torch::sqrt(torch::sum(r.pow(2), 1));
    torch::Tensor q = r / norm.unsqueeze(-1);

    using Slice = torch::indexing::Slice;
    torch::Tensor R = torch::zeros({q.size(0), 3, 3}, torch::device(torch::kCUDA));
    torch::Tensor r0 = q.index({Slice(), 0});
    torch::Tensor x = q.index({Slice(), 1});
    torch::Tensor y = q.index({Slice(), 2});
    torch::Tensor z = q.index({Slice(), 3});

    R.index_put_({Slice(), 0, 0}, 1 - 2 * (y * y + z * z));
    R.index_put_({Slice(), 0, 1}, 2 * (x * y - r0 * z));
    R.index_put_({Slice(), 0, 2}, 2 * (x * z + r0 * y));
    R.index_put_({Slice(), 1, 0}, 2 * (x * y + r0 * z));
    R.index_put_({Slice(), 1, 1}, 1 - 2 * (x * x + z * z));
    R.index_put_({Slice(), 1, 2}, 2 * (y * z - r0 * x));
    R.index_put_({Slice(), 2, 0}, 2 * (x * z - r0 * y));
    R.index_put_({Slice(), 2, 1}, 2 * (y * z + r0 * x));
    R.index_put_({Slice(), 2, 2}, 1 - 2 * (x * x + y * y));
    return R;
}

// Helper to compute covariance from quaternion and scale
static inline torch::Tensor quat_scale_to_covar(const torch::Tensor& quats,
                                                const torch::Tensor& scales) {
    auto rots = build_rotation(quats);
    auto scale_diag = torch::diag_embed(scales);
    return torch::bmm(torch::bmm(rots, scale_diag), rots.transpose(1, 2));
}

MCMC::MCMC(SplatData&& splat_data)
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

void MCMC::update_optimizer_for_relocate(torch::optim::Adam* optimizer,
                                         const torch::Tensor& sampled_indices,
                                         const torch::Tensor& dead_indices,
                                         int param_position) {
    // Get the parameter
    auto& param = optimizer->param_groups()[param_position].params()[0];
    void* param_key = param.unsafeGetTensorImpl();

    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(
        static_cast<torch::optim::AdamParamState&>(*optimizer->state()[param_key]));

    // Reset the states for sampled indices
    adamParamStates->exp_avg().index_put_({sampled_indices}, 0);
    adamParamStates->exp_avg_sq().index_put_({sampled_indices}, 0);

    // No need to update optimizer state since we're modifying in-place
}

void MCMC::update_optimizer_for_add(torch::optim::Adam* optimizer,
                                    const torch::Tensor& sampled_indices,
                                    int param_position) {
    void* param_key = optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl();

    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(
        static_cast<torch::optim::AdamParamState&>(*optimizer->state()[param_key]));

    optimizer->state().erase(param_key);

    // Extend the optimizer states
    int n_new = sampled_indices.size(0);
    auto device = adamParamStates->exp_avg().device();

    // Get the right shape for the new zeros based on the parameter dimension
    auto param_shape = optimizer->param_groups()[param_position].params()[0].sizes();
    std::vector<int64_t> new_shape;
    new_shape.push_back(n_new);
    for (int i = 1; i < param_shape.size(); ++i) {
        new_shape.push_back(param_shape[i]);
    }

    std::vector<torch::Tensor> exp_avg_tensors = {
        adamParamStates->exp_avg(),
        torch::zeros(new_shape, torch::TensorOptions().device(device))};
    std::vector<torch::Tensor> exp_avg_sq_tensors = {
        adamParamStates->exp_avg_sq(),
        torch::zeros(new_shape, torch::TensorOptions().device(device))};

    adamParamStates->exp_avg(torch::cat(exp_avg_tensors, 0));
    adamParamStates->exp_avg_sq(torch::cat(exp_avg_sq_tensors, 0));

    void* new_param_key = optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl();
    optimizer->state()[new_param_key] = std::move(adamParamStates);
}

int MCMC::relocate_gs() {
    auto opacities = _splat_data.get_opacity();
    auto dead_mask = opacities <= _min_opacity;
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
    int n_max = static_cast<int>(_binoms.size(0));
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
    new_opacities = torch::clamp(new_opacities, _min_opacity, 1.0f - 1e-7f);

    // Update parameters for sampled indices
    _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
    _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

    // Update optimizer states for sampled indices
    update_optimizer_for_relocate(_optimizer.get(), sampled_idxs, dead_indices, 5); // opacity

    // Copy from sampled to dead indices
    _splat_data.xyz().index_put_({dead_indices}, _splat_data.xyz().index_select(0, sampled_idxs));
    _splat_data.sh0().index_put_({dead_indices}, _splat_data.sh0().index_select(0, sampled_idxs));
    _splat_data.shN().index_put_({dead_indices}, _splat_data.shN().index_select(0, sampled_idxs));
    _splat_data.scaling_raw().index_put_({dead_indices}, _splat_data.scaling_raw().index_select(0, sampled_idxs));
    _splat_data.rotation_raw().index_put_({dead_indices}, _splat_data.rotation_raw().index_select(0, sampled_idxs));
    _splat_data.opacity_raw().index_put_({dead_indices}, _splat_data.opacity_raw().index_select(0, sampled_idxs));

    return n_dead;
}

int MCMC::add_new_gs() {
    int current_n = _splat_data.size();
    int n_target = std::min(_cap_max, static_cast<int>(1.05f * current_n));
    int n_new = std::max(0, n_target - current_n);

    if (n_new == 0)
        return 0;

    auto opacities = _splat_data.get_opacity();
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
    int n_max = static_cast<int>(_binoms.size(0));
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
    new_opacities = torch::clamp(new_opacities, _min_opacity, 1.0f - 1e-7f);

    // Update existing Gaussians
    _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
    _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

    // Concatenate new Gaussians
    auto new_xyz = _splat_data.xyz().index_select(0, sampled_idxs);
    auto new_sh0 = _splat_data.sh0().index_select(0, sampled_idxs);
    auto new_shN = _splat_data.shN().index_select(0, sampled_idxs);
    auto new_scaling = _splat_data.scaling_raw().index_select(0, sampled_idxs);
    auto new_rotation = _splat_data.rotation_raw().index_select(0, sampled_idxs);
    auto new_opacity = _splat_data.opacity_raw().index_select(0, sampled_idxs);

    // Update parameters
    _splat_data.xyz() = torch::cat({_splat_data.xyz(), new_xyz}, 0).set_requires_grad(true);
    _splat_data.sh0() = torch::cat({_splat_data.sh0(), new_sh0}, 0).set_requires_grad(true);
    _splat_data.shN() = torch::cat({_splat_data.shN(), new_shN}, 0).set_requires_grad(true);
    _splat_data.scaling_raw() = torch::cat({_splat_data.scaling_raw(), new_scaling}, 0).set_requires_grad(true);
    _splat_data.rotation_raw() = torch::cat({_splat_data.rotation_raw(), new_rotation}, 0).set_requires_grad(true);
    _splat_data.opacity_raw() = torch::cat({_splat_data.opacity_raw(), new_opacity}, 0).set_requires_grad(true);

    // Update optimizer states
    for (int i = 0; i < 6; ++i) {
        update_optimizer_for_add(_optimizer.get(), sampled_idxs, i);
        _optimizer->param_groups()[i].params()[0] =
            (i == 0) ? _splat_data.xyz() : (i == 1) ? _splat_data.sh0()
                                       : (i == 2)   ? _splat_data.shN()
                                       : (i == 3)   ? _splat_data.scaling_raw()
                                       : (i == 4)   ? _splat_data.rotation_raw()
                                                    : _splat_data.opacity_raw();
    }

    // Update max_radii2D
    _splat_data.max_radii2D() = torch::cat({_splat_data.max_radii2D(),
                                            torch::zeros({n_new}, torch::kFloat32).to(torch::kCUDA)},
                                           0);

    return n_new;
}

void MCMC::inject_noise() {
    auto opacities = _splat_data.get_opacity().flatten();
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

    // Generate noise
    auto noise = torch::randn_like(_splat_data.xyz()) * op_sigmoid.unsqueeze(-1) * _current_lr * _noise_lr;

    // Transform noise by covariance
    noise = torch::bmm(covars, noise.unsqueeze(-1)).squeeze(-1);

    // Add noise to positions
    _splat_data.xyz().add_(noise);
}

void MCMC::post_backward(int iter, gs::RenderOutput& render_output) {
    // Move binoms to device if needed
    _binoms = _binoms.to(_splat_data.xyz().device());

    // Refine Gaussians
    if (iter < _refine_stop_iter && iter > _refine_start_iter && iter % _refine_every == 0) {
        // Relocate dead Gaussians
        int n_relocated = relocate_gs();
        if (_verbose) {
            std::cout << "Step " << iter << ": Relocated " << n_relocated << " GSs." << std::endl;
        }

        // Add new Gaussians
        int n_added = add_new_gs();
        if (_verbose) {
            std::cout << "Step " << iter << ": Added " << n_added << " GSs. "
                      << "Now having " << _splat_data.size() << " GSs." << std::endl;
        }

        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    // Inject noise to positions
    inject_noise();
}

void MCMC::step(int iter) {
    if (iter < _params->iterations) {
        _optimizer->step();
        _optimizer->zero_grad(true);

        // Update learning rate (for noise injection)
        // Simple linear decay for now - can be replaced with more sophisticated scheduling
        float progress = static_cast<float>(iter) / _params->iterations;
        _current_lr = _params->position_lr_init * (1.0f - progress) +
                      _params->position_lr_final * progress;
    }
}

void MCMC::initialize(const gs::param::OptimizationParameters& optimParams) {
    _params = std::make_unique<gs::param::OptimizationParameters>(optimParams);
    const auto dev = torch::kCUDA;

    // Initialize parameters on CUDA
    _splat_data.xyz() = _splat_data.xyz().to(dev).set_requires_grad(true);
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

    // Initialize optimizer
    using torch::optim::AdamOptions;
    std::vector<torch::optim::OptimizerParamGroup> groups;

    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.xyz()},
                                                          std::make_unique<AdamOptions>(_params->position_lr_init * _splat_data.get_scene_scale())));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.sh0()},
                                                          std::make_unique<AdamOptions>(_params->feature_lr)));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.shN()},
                                                          std::make_unique<AdamOptions>(_params->feature_lr / 20.f)));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.scaling_raw()},
                                                          std::make_unique<AdamOptions>(_params->scaling_lr * _splat_data.get_scene_scale())));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.rotation_raw()},
                                                          std::make_unique<AdamOptions>(_params->rotation_lr)));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.opacity_raw()},
                                                          std::make_unique<AdamOptions>(_params->opacity_lr)));

    for (auto& g : groups)
        static_cast<AdamOptions&>(g.options()).eps(1e-15);

    _optimizer = std::make_unique<torch::optim::Adam>(groups, AdamOptions(0.f).eps(1e-15));

    // Initialize max_radii2D
    _splat_data.max_radii2D() = torch::zeros({_splat_data.size()}, torch::kFloat32).to(dev);

    // Initialize current learning rate
    _current_lr = _params->position_lr_init;
}