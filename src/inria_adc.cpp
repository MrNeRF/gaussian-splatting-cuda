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

void InriaADC::validate_tensor_sizes() {
    const int gs_count = _splat_data.size();

    TORCH_CHECK(_splat_data.xyz().size(0) == gs_count,
                "xyz size mismatch: ", _splat_data.xyz().size(0), " vs ", gs_count);
    TORCH_CHECK(_splat_data.sh0().size(0) == gs_count,
                "sh0 size mismatch: ", _splat_data.sh0().size(0), " vs ", gs_count);
    TORCH_CHECK(_splat_data.shN().size(0) == gs_count,
                "shN size mismatch: ", _splat_data.shN().size(0), " vs ", gs_count);
    TORCH_CHECK(_splat_data.scaling_raw().size(0) == gs_count,
                "scaling size mismatch: ", _splat_data.scaling_raw().size(0), " vs ", gs_count);
    TORCH_CHECK(_splat_data.rotation_raw().size(0) == gs_count,
                "rotation size mismatch: ", _splat_data.rotation_raw().size(0), " vs ", gs_count);
    TORCH_CHECK(_splat_data.opacity_raw().size(0) == gs_count,
                "opacity size mismatch: ", _splat_data.opacity_raw().size(0), " vs ", gs_count);
    TORCH_CHECK(_xyz_gradient_accum.size(0) == gs_count,
                "gradient accumulator size mismatch: ", _xyz_gradient_accum.size(0), " vs ", gs_count);
    TORCH_CHECK(_denom.size(0) == gs_count,
                "denom size mismatch: ", _denom.size(0), " vs ", gs_count);
    TORCH_CHECK(_splat_data.max_radii2D().size(0) == gs_count,
                "max_radii2D size mismatch: ", _splat_data.max_radii2D().size(0), " vs ", gs_count);
}

void InriaADC::Reset_opacity() {
    torch::cuda::synchronize();

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

    torch::cuda::synchronize();
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
    torch::cuda::synchronize();

    // Ensure mask has correct size
    TORCH_CHECK(mask.size(0) == _splat_data.size(),
                "Prune mask size (", mask.size(0), ") doesn't match gaussian count (",
                _splat_data.size(), ")");

    // reverse to keep points
    auto valid_point_mask = ~mask;
    auto indices = valid_point_mask.nonzero().squeeze(-1);

    if (indices.numel() == 0) {
        std::cerr << ts::color::YELLOW << "WARNING: Pruning would remove all gaussians!"
                  << ts::color::RESET << std::endl;
        return;
    }

    // Store original sizes for validation
    const int original_size = _splat_data.size();
    const int expected_new_size = indices.size(0);

    prune_optimizer(_optimizer.get(), indices, _splat_data.xyz(), 0);
    prune_optimizer(_optimizer.get(), indices, _splat_data.sh0(), 1);
    prune_optimizer(_optimizer.get(), indices, _splat_data.shN(), 2);
    prune_optimizer(_optimizer.get(), indices, _splat_data.scaling_raw(), 3);
    prune_optimizer(_optimizer.get(), indices, _splat_data.rotation_raw(), 4);
    prune_optimizer(_optimizer.get(), indices, _splat_data.opacity_raw(), 5);

    // Index select on 1D tensors
    _xyz_gradient_accum = _xyz_gradient_accum.index_select(0, indices);
    _denom = _denom.index_select(0, indices);
    _splat_data.max_radii2D() = _splat_data.max_radii2D().index_select(0, indices);

    torch::cuda::synchronize();

    // Validate new size
    TORCH_CHECK(_splat_data.size() == expected_new_size,
                "After pruning, size mismatch: expected ", expected_new_size,
                " but got ", _splat_data.size());

    validate_tensor_sizes();
}

void cat_tensors_to_optimizer(torch::optim::Adam* optimizer,
                              torch::Tensor& extension_tensor,
                              torch::Tensor& old_tensor,
                              int param_position) {
    torch::cuda::synchronize();

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

    torch::cuda::synchronize();
}

void InriaADC::densification_postfix(torch::Tensor& new_xyz,
                                     torch::Tensor& new_sh0,
                                     torch::Tensor& new_shN,
                                     torch::Tensor& new_scaling,
                                     torch::Tensor& new_rotation,
                                     torch::Tensor& new_opacity) {
    // Ensure all new tensors have the same size
    const int n_new = new_xyz.size(0);
    TORCH_CHECK(new_sh0.size(0) == n_new, "new_sh0 size mismatch");
    TORCH_CHECK(new_shN.size(0) == n_new, "new_shN size mismatch");
    TORCH_CHECK(new_scaling.size(0) == n_new, "new_scaling size mismatch");
    TORCH_CHECK(new_rotation.size(0) == n_new, "new_rotation size mismatch");
    TORCH_CHECK(new_opacity.size(0) == n_new, "new_opacity size mismatch");

    const int old_size = _splat_data.size();

    cat_tensors_to_optimizer(_optimizer.get(), new_xyz, _splat_data.xyz(), 0);
    cat_tensors_to_optimizer(_optimizer.get(), new_sh0, _splat_data.sh0(), 1);
    cat_tensors_to_optimizer(_optimizer.get(), new_shN, _splat_data.shN(), 2);
    cat_tensors_to_optimizer(_optimizer.get(), new_scaling, _splat_data.scaling_raw(), 3);
    cat_tensors_to_optimizer(_optimizer.get(), new_rotation, _splat_data.rotation_raw(), 4);
    cat_tensors_to_optimizer(_optimizer.get(), new_opacity, _splat_data.opacity_raw(), 5);

    // CRITICAL: Synchronize CUDA before resetting accumulators
    torch::cuda::synchronize();

    const int new_size = _splat_data.size();
    TORCH_CHECK(new_size == old_size + n_new,
                "Size after concatenation incorrect: expected ", old_size + n_new,
                " but got ", new_size);

    // Reset accumulators with correct 1D shape
    _xyz_gradient_accum = torch::zeros({new_size}, torch::kFloat32).to(torch::kCUDA);
    _denom = torch::zeros({new_size}, torch::kFloat32).to(torch::kCUDA);
    _splat_data.max_radii2D() = torch::zeros({new_size}, torch::kFloat32).to(torch::kCUDA);

    validate_tensor_sizes();
}

void InriaADC::densify_and_split(torch::Tensor& grads, float grad_threshold, float min_opacity,
                                 int current_model_size, int gaussians_added_by_clone) {
    static const int N = 2;

    // FIXED: Use gradient size which represents the original model size
    const int n_init_points = grads.size(0);

    // grads is [N], not [N, 1]
    torch::Tensor selected_pts_mask = grads >= grad_threshold;
    TORCH_CHECK(selected_pts_mask.size(0) == n_init_points,
                "selected_pts_mask size mismatch: ", selected_pts_mask.size(0), " vs ", n_init_points);

    // Check scale condition for splitting (large gaussians)
    // FIXED: Only check the first n_init_points gaussians
    auto scaling = _splat_data.get_scaling().index({torch::indexing::Slice(0, n_init_points)});
    TORCH_CHECK(scaling.size(0) == n_init_points,
                "scaling size mismatch: ", scaling.size(0), " vs ", n_init_points);

    auto scale_mask = std::get<0>(scaling.max(1)) > _percent_dense * _splat_data.get_scene_scale();
    TORCH_CHECK(scale_mask.size(0) == n_init_points,
                "scale_mask size mismatch: ", scale_mask.size(0), " vs ", n_init_points);

    // Combine conditions
    selected_pts_mask = selected_pts_mask & scale_mask;

    auto indices = selected_pts_mask.nonzero().squeeze(-1);

    if (indices.numel() == 0) {
        return;  // No points to split
    }

    // Prepare new gaussians
    torch::Tensor stds = _splat_data.get_scaling().index_select(0, indices).repeat({N, 1});
    torch::Tensor means = torch::zeros({stds.size(0), 3}).to(torch::kCUDA);
    torch::Tensor samples = torch::randn({stds.size(0), stds.size(1)}).to(torch::kCUDA) * stds + means;
    torch::Tensor rots = build_rotation(_splat_data.rotation_raw().index_select(0, indices)).repeat({N, 1, 1});

    torch::Tensor new_xyz = torch::bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + _splat_data.xyz().index_select(0, indices).repeat({N, 1});
    torch::Tensor new_scaling = torch::log(_splat_data.get_scaling().index_select(0, indices).repeat({N, 1}) / (0.8 * N));
    torch::Tensor new_rotation = _splat_data.rotation_raw().index_select(0, indices).repeat({N, 1});
    torch::Tensor new_sh0 = _splat_data.sh0().index_select(0, indices).repeat({N, 1, 1});
    torch::Tensor new_shN = _splat_data.shN().index_select(0, indices).repeat({N, 1, 1});
    torch::Tensor new_opacity = _splat_data.opacity_raw().index_select(0, indices).repeat({N, 1});

    // Add new gaussians
    densification_postfix(new_xyz, new_sh0, new_shN, new_scaling, new_rotation, new_opacity);

    torch::cuda::synchronize();

    // FIXED: Calculate sizes correctly accounting for clone operations
    const int size_after_split = _splat_data.size();
    const int n_new_gaussians_from_split = indices.numel() * N;
    const int total_new_gaussians = size_after_split - n_init_points;

    // Verify the math
    TORCH_CHECK(total_new_gaussians == gaussians_added_by_clone + n_new_gaussians_from_split,
                "Unexpected number of new gaussians: expected ",
                gaussians_added_by_clone + n_new_gaussians_from_split,
                " but got ", total_new_gaussians);

    // Create prune filter for original gaussians (before adding new ones)
    torch::Tensor prune_filter = torch::zeros({n_init_points}, torch::kBool).to(torch::kCUDA);
    prune_filter.masked_scatter_(selected_pts_mask, torch::ones_like(selected_pts_mask, torch::kBool));

    // Extend prune filter to include all newly added gaussians (from both clone and split)
    torch::Tensor extended_prune_filter = torch::cat({
        prune_filter,
        torch::zeros({total_new_gaussians}, torch::kBool).to(torch::kCUDA)
    });

    TORCH_CHECK(extended_prune_filter.size(0) == size_after_split,
                "Extended prune filter size mismatch: ", extended_prune_filter.size(0),
                " vs ", size_after_split);

    // Also prune low opacity gaussians
    auto opacity_check = _splat_data.get_opacity();
    if (opacity_check.dim() == 2 && opacity_check.size(1) == 1) {
        opacity_check = opacity_check.squeeze(-1);
    }
    TORCH_CHECK(opacity_check.size(0) == size_after_split,
                "Opacity size mismatch: ", opacity_check.size(0), " vs ", size_after_split);

    extended_prune_filter = extended_prune_filter | (opacity_check < min_opacity);

    prune_points(extended_prune_filter);
}

void InriaADC::densify_and_clone(torch::Tensor& grads, float grad_threshold) {
    // FIXED: Use gradient size, not current model size
    const int n_init_points = grads.size(0);

    // grads is [N]
    torch::Tensor selected_pts_mask = grads >= grad_threshold;

    // Check scale condition for cloning (small gaussians)
    // FIXED: Only check the first n_init_points gaussians
    auto scaling = _splat_data.get_scaling().index({torch::indexing::Slice(0, n_init_points)});
    TORCH_CHECK(scaling.size(0) == n_init_points,
                "scaling size mismatch: ", scaling.size(0), " vs ", n_init_points);

    auto scale_mask = std::get<0>(scaling.max(1)) <= _percent_dense * _splat_data.get_scene_scale();

    // Combine conditions
    selected_pts_mask = selected_pts_mask & scale_mask;

    auto indices = selected_pts_mask.nonzero().squeeze(-1);

    if (indices.numel() == 0) {
        return;  // No points to clone
    }

    torch::Tensor new_xyz = _splat_data.xyz().index_select(0, indices);
    torch::Tensor new_sh0 = _splat_data.sh0().index_select(0, indices);
    torch::Tensor new_shN = _splat_data.shN().index_select(0, indices);
    torch::Tensor new_opacity = _splat_data.opacity_raw().index_select(0, indices);
    torch::Tensor new_scaling = _splat_data.scaling_raw().index_select(0, indices);
    torch::Tensor new_rotation = _splat_data.rotation_raw().index_select(0, indices);

    densification_postfix(new_xyz, new_sh0, new_shN, new_scaling, new_rotation, new_opacity);
}

void InriaADC::Densify_and_prune(float max_grad, float min_opacity) {
    torch::cuda::synchronize();
    validate_tensor_sizes();

    // ADDED: Extra validation to ensure accumulator matches model size
    const int model_size = _splat_data.size();
    if (_xyz_gradient_accum.size(0) != model_size || _denom.size(0) != model_size) {
        std::cerr << ts::color::YELLOW << "WARNING: Gradient accumulator size mismatch. "
                  << "Resetting accumulators." << ts::color::RESET << std::endl;
        _xyz_gradient_accum = torch::zeros({model_size}, torch::kFloat32).to(torch::kCUDA);
        _denom = torch::zeros({model_size}, torch::kFloat32).to(torch::kCUDA);
        return; // Skip densification this iteration
    }

    // Average gradients by count
    torch::Tensor grads = _xyz_gradient_accum / _denom.clamp_min(1.0);
    grads.masked_fill_(grads.isnan(), 0.0);

    TORCH_CHECK(grads.size(0) == _splat_data.size(),
                "Gradient size mismatch: ", grads.size(0), " vs ", _splat_data.size());

    // FIXED: Track model size between operations
    const int size_before_clone = _splat_data.size();
    densify_and_clone(grads, max_grad);

    const int size_after_clone = _splat_data.size();
    const int gaussians_added_by_clone = size_after_clone - size_before_clone;

    // Now call split with awareness of the clone additions
    densify_and_split(grads, max_grad, min_opacity, size_after_clone, gaussians_added_by_clone);

    // CRITICAL: Reset gradient statistics after densification
    // This matches the Python implementation
    torch::cuda::synchronize();
    const int current_size = _splat_data.size();
    _xyz_gradient_accum = torch::zeros({current_size}, torch::kFloat32).to(torch::kCUDA);
    _denom = torch::zeros({current_size}, torch::kFloat32).to(torch::kCUDA);

    validate_tensor_sizes();
}

void InriaADC::Add_densification_stats(torch::Tensor& viewspace_point_tensor,
                                       torch::Tensor& update_filter,
                                       int width,
                                       int height,
                                       int n_cameras) {
    // Check if gradient exists
    if (!viewspace_point_tensor.grad().defined()) {
        std::cerr << ts::color::RED << "ERROR: viewspace_point_tensor has no gradient! "
                  << "Make sure to call retain_grad() on intermediate tensors."
                  << ts::color::RESET << std::endl;
        return;
    }

    // FIXED: Use actual model size instead of viewspace tensor size
    const int current_model_size = _splat_data.size();
    const int viewspace_size = viewspace_point_tensor.size(0);
    const int accumulator_size = _xyz_gradient_accum.size(0);

    // Check if there's a size mismatch
    if (viewspace_size != current_model_size) {
        std::cerr << ts::color::YELLOW << "WARNING: Viewspace tensor size (" << viewspace_size
                  << ") doesn't match model size (" << current_model_size << ")"
                  << ts::color::RESET << std::endl;
        // Skip this gradient accumulation to avoid errors
        return;
    }

    if (current_model_size != accumulator_size) {
        std::cerr << ts::color::YELLOW << "WARNING: Resizing gradient accumulators from "
                  << accumulator_size << " to " << current_model_size << ts::color::RESET << std::endl;
        _xyz_gradient_accum = torch::zeros({current_model_size}, torch::kFloat32).to(torch::kCUDA);
        _denom = torch::zeros({current_model_size}, torch::kFloat32).to(torch::kCUDA);
    }

    // Clone the gradient to avoid modifying the original
    auto grad = viewspace_point_tensor.grad().clone();

    // CRITICAL: Normalize gradients to [-1, 1] screen space
    // This matches the Python implementation in default.py
    grad.index({torch::indexing::Slice(), 0}) *= (width / 2.0f) * n_cameras;
    grad.index({torch::indexing::Slice(), 1}) *= (height / 2.0f) * n_cameras;

    // Get indices where update_filter is true
    auto indices = update_filter.nonzero().squeeze(-1);
    if (indices.numel() == 0) {
        return; // No points to update
    }

    // Validate indices are within bounds
    auto max_idx = indices.max().item<int64_t>();
    if (max_idx >= _xyz_gradient_accum.size(0)) {
        std::cerr << ts::color::RED << "ERROR: Index " << max_idx
                  << " out of bounds for accumulator size " << _xyz_gradient_accum.size(0)
                  << ts::color::RESET << std::endl;
        return;
    }

    // Compute L2 norm of gradients for visible points
    auto selected_grad = grad.index_select(0, indices);
    auto grad_norm = selected_grad.norm(2, -1);  // L2 norm, result is 1D

    // Accumulate gradient norms using index_add_ for efficiency
    _xyz_gradient_accum.index_add_(0, indices, grad_norm);

    // Accumulate count
    _denom.index_add_(0, indices, torch::ones_like(indices, torch::kFloat32));
}

void InriaADC::post_backward(int iter, RenderOutput& render_output) {
    // CRITICAL: Move SH degree increment to BEFORE any densification logic
    // to avoid race conditions during densification
    bool sh_degree_changed = false;
    if (iter % 1000 == 0 && iter > 0) {
        torch::cuda::synchronize();
        _splat_data.increment_sh_degree();
        sh_degree_changed = true;
        torch::cuda::synchronize();
    }

    // Skip the rest if we just changed SH degree to avoid any size mismatches
    if (sh_degree_changed) {
        return;
    }

    // Update max radii for visible Gaussians
    auto visibility = render_output.visibility;
    if (visibility.any().item<bool>()) {
        // Ensure max_radii2D has correct size
        if (_splat_data.max_radii2D().size(0) != visibility.size(0)) {
            std::cerr << ts::color::YELLOW << "WARNING: Resizing max_radii2D from "
                      << _splat_data.max_radii2D().size(0) << " to " << visibility.size(0)
                      << ts::color::RESET << std::endl;
            _splat_data.max_radii2D() = torch::zeros({visibility.size(0)}, torch::kFloat32).to(torch::kCUDA);
        }

        // Get current max radii for visible points
        auto visible_max_radii = _splat_data.max_radii2D().masked_select(visibility);

        // Get rendered radii for visible points
        auto visible_radii = render_output.radii.masked_select(visibility);

        // Update max radii with the maximum of current and rendered
        auto new_max_radii = torch::max(visible_max_radii, visible_radii);
        _splat_data.max_radii2D().masked_scatter_(visibility, new_max_radii);
    }

    // Densification & pruning logic
    if (iter < _params->densify_until_iter) {
        // Accumulate densification statistics
        Add_densification_stats(render_output.viewspace_pts,
                                visibility,
                                render_output.width,
                                render_output.height,
                                render_output.n_cameras);

        // Check if we should densify at this iteration
        const bool should_densify = (iter >= _params->densify_from_iter &&
                                     iter % _params->densification_interval == 0);

        if (should_densify) {
            // Get the current size before densification
            int size_before = _splat_data.size();

            // Perform densification and pruning
            Densify_and_prune(_params->densify_grad_threshold, _params->min_opacity);

            // Log densification results
            int size_after = _splat_data.size();
            if (size_after != size_before) {
                std::cout << ts::color::GREEN << "[Iter " << iter << "] "
                          << "Densified: " << size_before << " -> " << size_after
                          << " gaussians" << ts::color::RESET << std::endl;
            }
        }

        // Reset opacity periodically
        if (iter % _params->opacity_reset_interval == 0 &&
            iter > _params->densify_from_iter) {
            std::cout << ts::color::YELLOW << "[Iter " << iter << "] "
                      << "Resetting opacity" << ts::color::RESET << std::endl;
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
    _splat_data.sh0() = _splat_data.sh0().to(dev).set_requires_grad(true);
    _splat_data.shN() = _splat_data.shN().to(dev).set_requires_grad(true);

    // aux buffers (no grad) - Initialize as 1D tensors
    _percent_dense = _params->percent_dense;
    _xyz_gradient_accum = torch::zeros({_splat_data.size()}, torch::kFloat32).to(dev);
    _denom = torch::zeros({_splat_data.size()}, torch::kFloat32).to(dev);
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

    _optimizer = std::make_unique<torch::optim::Adam>(
        groups, AdamOptions(0.f).eps(1e-15));

    // Validate initial state
    validate_tensor_sizes();
}