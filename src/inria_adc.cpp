#include "core/inria_adc.hpp"
#include "core/debug_utils.hpp"
#include "core/parameters.hpp"
#include "core/render_utils.hpp"
#include <exception>
#include <thread>

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

static inline torch::Tensor inverse_sigmoid(torch::Tensor x) {
    return torch::log(x / (1 - x));
}

float InriaADC::Expon_lr_func::operator()(int64_t step) const {
    if (step < 0 || (lr_init == 0.0 && lr_final == 0.0)) {
        return 0.0;
    }
    float delay_rate;
    if (lr_delay_steps > 0. && step != 0) {
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * std::sin(0.5 * M_PI * std::clamp((float)step / lr_delay_steps, 0.f, 1.f));
    } else {
        delay_rate = 1.0;
    }
    float t = std::clamp(static_cast<float>(step) / static_cast<float>(max_steps), 0.f, 1.f);
    float log_lerp = std::exp(std::log(lr_init) * (1.f - t) + std::log(lr_final) * t);
    return delay_rate * log_lerp;
}

InriaADC::InriaADC(SplatData&& splat_data)
    : _splat_data(std::move(splat_data)) {
}

void InriaADC::Update_learning_rate(float iteration) {
    // This is hacky because you cant change in libtorch individual parameter learning rate
    // xyz is added first, since _optimizer->param_groups() return a vector, we assume that xyz stays first
    auto lr = _xyz_scheduler_args(iteration);
    static_cast<torch::optim::AdamOptions&>(_optimizer->param_groups()[0].options()).set_lr(lr);
}

void InriaADC::Reset_opacity() {
    // opacity activation
    auto new_opacity = inverse_sigmoid(torch::ones_like(_splat_data.opacity_raw(), torch::TensorOptions().dtype(torch::kFloat32)) * 0.01f);

    void* param_key = _optimizer->param_groups()[5].params()[0].unsafeGetTensorImpl();

    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *_optimizer->state()[param_key]));

    _optimizer->state().erase(param_key);

    adamParamStates->exp_avg(torch::zeros_like(new_opacity));
    adamParamStates->exp_avg_sq(torch::zeros_like(new_opacity));
    // replace tensor
    _optimizer->param_groups()[5].params()[0] = new_opacity.set_requires_grad(true);
    _splat_data.opacity_raw() = _optimizer->param_groups()[5].params()[0];

    void* new_param_key = _optimizer->param_groups()[5].params()[0].unsafeGetTensorImpl();
    _optimizer->state()[new_param_key] = std::move(adamParamStates);
}

void prune_optimizer(torch::optim::Adam* optimizer, const torch::Tensor& mask, torch::Tensor& old_tensor, int param_position) {
    void* param_key = optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl();

    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *optimizer->state()[param_key]));
    optimizer->state().erase(param_key);

    adamParamStates->exp_avg(adamParamStates->exp_avg().index_select(0, mask));
    adamParamStates->exp_avg_sq(adamParamStates->exp_avg_sq().index_select(0, mask));

    optimizer->param_groups()[param_position].params()[0] = old_tensor.index_select(0, mask).set_requires_grad(true);
    old_tensor = optimizer->param_groups()[param_position].params()[0]; // update old tensor

    void* new_param_key = optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl();
    optimizer->state()[new_param_key] = std::move(adamParamStates);
}

void InriaADC::prune_points(torch::Tensor mask) {
    // reverse to keep points
    auto valid_point_mask = ~mask;
    int true_count = valid_point_mask.sum().item<int>();
    auto indices = torch::nonzero(valid_point_mask == true).squeeze(-1);
    prune_optimizer(_optimizer.get(), indices, _splat_data.xyz(), 0);
    prune_optimizer(_optimizer.get(), indices, _splat_data.features_dc(), 1);
    prune_optimizer(_optimizer.get(), indices, _splat_data.features_rest(), 2);
    prune_optimizer(_optimizer.get(), indices, _splat_data.scaling_raw(), 3);
    prune_optimizer(_optimizer.get(), indices, _splat_data.rotation_raw(), 4);
    prune_optimizer(_optimizer.get(), indices, _splat_data.opacity_raw(), 5);

    _xyz_gradient_accum = _xyz_gradient_accum.index_select(0, indices);
    _denom = _denom.index_select(0, indices);
    _splat_data.max_radii2D() = _splat_data.max_radii2D().index_select(0, indices);
}

void cat_tensors_to_optimizer(torch::optim::Adam* optimizer,
                              torch::Tensor& extension_tensor,
                              torch::Tensor& old_tensor,
                              int param_position) {
    void* param_key = optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl();

    auto adamParamStates = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState&>(
        *optimizer->state()[param_key]));
    optimizer->state().erase(param_key);

    std::vector<torch::Tensor> exp_avg_tensors = {adamParamStates->exp_avg(), torch::zeros_like(extension_tensor)};
    std::vector<torch::Tensor> exp_avg_sq_tensors = {adamParamStates->exp_avg_sq(), torch::zeros_like(extension_tensor)};
    std::vector<torch::Tensor> param_tensors = {old_tensor, extension_tensor};

    adamParamStates->exp_avg(torch::cat(exp_avg_tensors, 0));
    adamParamStates->exp_avg_sq(torch::cat(exp_avg_sq_tensors, 0));

    optimizer->param_groups()[param_position].params()[0] = torch::cat(param_tensors, 0).set_requires_grad(true);
    old_tensor = optimizer->param_groups()[param_position].params()[0];

    void* new_param_key = optimizer->param_groups()[param_position].params()[0].unsafeGetTensorImpl();
    optimizer->state()[new_param_key] = std::move(adamParamStates);
}

void InriaADC::densification_postfix(torch::Tensor& new_xyz,
                                     torch::Tensor& new_features_dc,
                                     torch::Tensor& new_features_rest,
                                     torch::Tensor& new_scaling,
                                     torch::Tensor& new_rotation,
                                     torch::Tensor& new_opacity) {
    cat_tensors_to_optimizer(_optimizer.get(), new_xyz, _splat_data.xyz(), 0);
    cat_tensors_to_optimizer(_optimizer.get(), new_features_dc, _splat_data.features_dc(), 1);
    cat_tensors_to_optimizer(_optimizer.get(), new_features_rest, _splat_data.features_rest(), 2);
    cat_tensors_to_optimizer(_optimizer.get(), new_scaling, _splat_data.scaling_raw(), 3);
    cat_tensors_to_optimizer(_optimizer.get(), new_rotation, _splat_data.rotation_raw(), 4);
    cat_tensors_to_optimizer(_optimizer.get(), new_opacity, _splat_data.opacity_raw(), 5);

    _xyz_gradient_accum = torch::zeros({_splat_data.size(), 1}).to(torch::kCUDA);
    _denom = torch::zeros({_splat_data.size(), 1}).to(torch::kCUDA);
    _splat_data.max_radii2D() = torch::zeros({_splat_data.size()}).to(torch::kCUDA);
}

void InriaADC::densify_and_split(torch::Tensor& grads, float grad_threshold, float min_opacity) {
    static const int N = 2;
    const int n_init_points = _splat_data.size();
    // Extract points that satisfy the gradient condition
    torch::Tensor padded_grad = torch::zeros({n_init_points}).to(torch::kCUDA);
    padded_grad.slice(0, 0, grads.size(0)) = grads.squeeze();
    torch::Tensor selected_pts_mask = torch::where(padded_grad >= grad_threshold, torch::ones_like(padded_grad).to(torch::kBool), torch::zeros_like(padded_grad).to(torch::kBool));
    selected_pts_mask = torch::logical_and(selected_pts_mask, std::get<0>(_splat_data.get_scaling().max(1)) > _percent_dense * _splat_data.get_scene_scale());
    auto indices = torch::nonzero(selected_pts_mask == true).squeeze(-1);

    torch::Tensor stds = _splat_data.get_scaling().index_select(0, indices).repeat({N, 1});
    torch::Tensor means = torch::zeros({stds.size(0), 3}).to(torch::kCUDA);
    torch::Tensor samples = torch::randn({stds.size(0), stds.size(1)}).to(torch::kCUDA) * stds + means;
    torch::Tensor rots = build_rotation(_splat_data.rotation_raw().index_select(0, indices)).repeat({N, 1, 1});

    torch::Tensor new_xyz = torch::bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + _splat_data.xyz().index_select(0, indices).repeat({N, 1});
    torch::Tensor new_scaling = torch::log(_splat_data.get_scaling().index_select(0, indices).repeat({N, 1}) / (0.8 * N));
    torch::Tensor new_rotation = _splat_data.rotation_raw().index_select(0, indices).repeat({N, 1});
    torch::Tensor new_features_dc = _splat_data.features_dc().index_select(0, indices).repeat({N, 1, 1});
    torch::Tensor new_features_rest = _splat_data.features_rest().index_select(0, indices).repeat({N, 1, 1});
    torch::Tensor new_opacity = _splat_data.opacity_raw().index_select(0, indices).repeat({N, 1});

    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity);

    torch::Tensor prune_filter = torch::cat({selected_pts_mask, torch::zeros({N * selected_pts_mask.sum().item<int>()}).to(torch::kBool).to(torch::kCUDA)});
    prune_filter = torch::logical_or(prune_filter, (_splat_data.get_opacity() < min_opacity).squeeze(-1));
    prune_points(prune_filter);
}

void InriaADC::densify_and_clone(torch::Tensor& grads, float grad_threshold) {
    // Extract points that satisfy the gradient condition
    torch::Tensor selected_pts_mask = torch::where(grads >= grad_threshold,
                                                   torch::ones_like(grads).to(torch::kBool),
                                                   torch::zeros_like(grads).to(torch::kBool));

    selected_pts_mask = torch::logical_and(selected_pts_mask, std::get<0>(_splat_data.get_scaling().max(1)).unsqueeze(-1) <= _percent_dense * _splat_data.get_scene_scale());

    auto indices = torch::nonzero(selected_pts_mask.squeeze(-1) == true).squeeze(-1);
    torch::Tensor new_xyz = _splat_data.xyz().index_select(0, indices);
    torch::Tensor new_features_dc = _splat_data.features_dc().index_select(0, indices);
    torch::Tensor new_features_rest = _splat_data.features_rest().index_select(0, indices);
    torch::Tensor new_opacity = _splat_data.opacity_raw().index_select(0, indices);
    torch::Tensor new_scaling = _splat_data.scaling_raw().index_select(0, indices);
    torch::Tensor new_rotation = _splat_data.rotation_raw().index_select(0, indices);

    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity);
}

void InriaADC::Densify_and_prune(float max_grad, float min_opacity) {
    torch::Tensor grads = _xyz_gradient_accum / _denom;
    grads.index_put_({grads.isnan()}, 0.0);

    densify_and_clone(grads, max_grad);
    densify_and_split(grads, max_grad, min_opacity);
}

void InriaADC::Add_densification_stats(torch::Tensor& viewspace_point_tensor, torch::Tensor& update_filter) {
    _xyz_gradient_accum.index_put_({update_filter}, _xyz_gradient_accum.index_select(0, update_filter.nonzero().squeeze()) + viewspace_point_tensor.grad().index_select(0, update_filter.nonzero().squeeze()).slice(1, 0, 2).norm(2, -1, true));
    _denom.index_put_({update_filter}, _denom.index_select(0, update_filter.nonzero().squeeze()) + 1);
}

void InriaADC::post_backward(int iter, RenderOutput& render_output) {

    if (iter % 1000 == 0)
        _splat_data.increment_sh_degree();

    const auto visible_max_radii = _splat_data.max_radii2D().masked_select(render_output.visibility);
    const auto visible_radii = render_output.radii.masked_select(render_output.visibility);
    _splat_data.max_radii2D().masked_scatter_(render_output.visibility, torch::max(visible_max_radii, visible_radii));

    // Densification & pruning
    if (iter < _params->densify_until_iter) {
        Add_densification_stats(render_output.viewspace_pts, render_output.visibility);
        const bool is_densifying = (iter < _params->densify_until_iter &&
                                    iter > _params->densify_from_iter &&
                                    iter % _params->densification_interval == 0);
        if (is_densifying) {
            Densify_and_prune(_params->densify_grad_threshold, _params->min_opacity);
        }
        if (iter % _params->opacity_reset_interval == 0) {
            Reset_opacity();
        }
    }
}

void InriaADC::step(int iter) {

    if (iter < _params->iterations) {
        _optimizer->step();
        _optimizer->zero_grad(true);
        Update_learning_rate(iter);
    }
}

void InriaADC::initialize(const gs::param::OptimizationParameters& optimParams) {
    _params = std::make_unique<gs::param::OptimizationParameters>(optimParams);
    const auto dev = torch::kCUDA;

    _splat_data.xyz() = _splat_data.xyz().to(dev).set_requires_grad(true);
    _splat_data.scaling_raw() = _splat_data.scaling_raw().to(dev).set_requires_grad(true);
    _splat_data.rotation_raw() = _splat_data.rotation_raw().to(dev).set_requires_grad(true);
    _splat_data.opacity_raw() = _splat_data.opacity_raw().to(dev).set_requires_grad(true);
    _splat_data.features_dc() = _splat_data.features_dc().to(dev).set_requires_grad(true);
    _splat_data.features_rest() = _splat_data.features_rest().to(dev).set_requires_grad(true);

    // aux buffers (no grad)
    _percent_dense = _params->percent_dense;
    _xyz_gradient_accum = torch::zeros({_splat_data.size(), 1}, torch::kFloat32).to(dev);
    _denom = torch::zeros({_splat_data.size(), 1}, torch::kFloat32).to(dev);
    _splat_data.max_radii2D() = torch::zeros({_splat_data.size()}, torch::kFloat32).to(dev);

    _xyz_scheduler_args = Expon_lr_func(
        _params->position_lr_init * _splat_data.get_scene_scale(),
        _params->position_lr_final * _splat_data.get_scene_scale(),
        _params->position_lr_delay_mult,
        _params->position_lr_max_steps);

    // ------------------------------------------------------------------
    // 1. Build Adam param-groups with the CUDA tensors
    // ------------------------------------------------------------------
    using torch::optim::AdamOptions;
    std::vector<torch::optim::OptimizerParamGroup> groups;

    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.xyz()},
                                                          std::make_unique<AdamOptions>(_params->position_lr_init * _splat_data.get_scene_scale())));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.features_dc()},
                                                          std::make_unique<AdamOptions>(_params->feature_lr)));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.features_rest()},
                                                          std::make_unique<AdamOptions>(_params->feature_lr / 20.f)));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.scaling_raw()},
                                                          std::make_unique<AdamOptions>(_params->scaling_lr * _splat_data.get_scene_scale())));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.rotation_raw()},
                                                          std::make_unique<AdamOptions>(_params->rotation_lr)));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.opacity_raw()},
                                                          std::make_unique<AdamOptions>(_params->opacity_lr)));

    for (auto& g : groups)
        static_cast<AdamOptions&>(g.options()).eps(1e-15);

    _optimizer = std::make_unique<torch::optim::Adam>(
        groups, AdamOptions(0.f).eps(1e-15));
}