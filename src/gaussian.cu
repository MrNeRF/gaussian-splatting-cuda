#include "gaussian.cuh"
#include <exception>

GaussianModel::GaussianModel(int sh_degree) : _max_sh_degree(sh_degree),
                                              _active_sh_degree(0),
                                              _xyz_scheduler_args(Expon_lr_func(0.0, 1.0)) { // this is really ugly

    _xyz = torch::empty({0});
    _features_dc = torch::empty({0});
    _features_rest = torch::empty({0});
    _scaling = torch::empty({0});
    _rotation = torch::empty({0});
    _opacity = torch::empty({0});
    _max_radii2D = torch::empty({0});
    _xyz_gradient_accum = torch::empty({0});

    // IMPORTANT: The order has to stay fix because we must now the place of the parameters
    // The optimizer just gets the tensor. There is currently no way to access the parameters by name
    register_parameter("xyz", _xyz, true);
    register_parameter("features_dc", _features_dc, true);
    register_parameter("features_rest", _features_rest, true);
    register_parameter("scaling", _scaling, true);
    register_parameter("rotation", _rotation, true);
    register_parameter("opacity", _opacity, true);
}

torch::Tensor GaussianModel::Get_covariance(float scaling_modifier) {
    auto L = build_scaling_rotation(scaling_modifier * Get_scaling(), _rotation);
    auto actual_covariance = torch::mm(L, L.transpose(1, 2));
    auto symm = strip_symmetric(actual_covariance);
    return symm;
}

/**
 * @brief Fetches the features of the Gaussian model
 *
 * This function concatenates _features_dc and _features_rest along the second dimension.
 *
 * @return Tensor of the concatenated features
 */
torch::Tensor GaussianModel::Get_features() const {
    auto features_dc = _features_dc;
    auto features_rest = _features_rest;
    return torch::cat({features_dc, features_rest}, 1);
}

/**
 * @brief Increment the SH degree by 1
 *
 * This function increments the active_sh_degree by 1, up to a maximum of max_sh_degree.
 */
void GaussianModel::One_up_sh_degree() {
    if (_active_sh_degree < _max_sh_degree) {
        _active_sh_degree++;
    }
}

/**
 * @brief Initialize Gaussian Model from a Point Cloud.
 *
 * This function creates a Gaussian model from a given PointCloud object. It also sets
 * the spatial learning rate scale. The model's features, scales, rotations, and opacities
 * are initialized based on the input point cloud.
 *
 * @param pcd The input point cloud
 * @param spatial_lr_scale The spatial learning rate scale
 */
void GaussianModel::Create_from_pcd(PointCloud& pcd, float spatial_lr_scale) {
    std::cout << "Creating from pcd" << std::endl;
    _spatial_lr_scale = spatial_lr_scale;
    //  load points
    auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor fused_point_cloud = torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size()), 3}, pointType).to(torch::kCUDA);

    // load colors
    auto colorType = torch::TensorOptions().dtype(torch::kUInt8);
    auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size()), 3}, colorType).to(torch::kCUDA));

    auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))}).to(torch::kCUDA);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None), torch::indexing::Slice(1, torch::indexing::None)}, 0.0);

    std::cout << "Number of points at initialisation : " << fused_point_cloud.size(0) << std::endl;

    auto dist2 = torch::clamp_min(distCUDA2(torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size()), 3}, pointType).to(torch::kCUDA)), 0.0000001);

    auto scales = torch::log(torch::sqrt(dist2)).unsqueeze(-1).repeat({1, 3}, 0);
    auto rots = torch::zeros({fused_point_cloud.size(0), 4}).to(torch::kCUDA);
    rots.index_put_({torch::indexing::Slice(), 0}, 1);
    auto opacities = inverse_sigmoid(0.5 * torch::ones({fused_point_cloud.size(0), 1}).to(torch::kCUDA));
    _xyz = fused_point_cloud.set_requires_grad(true);
    std::cout << "features size before transpose: (" << features.size(0) << ", " << features.size(1) << ", " << features.size(2) << ")" << std::endl;
    _features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().set_requires_grad(true);
    std::cout << "_features_dc size after transpose: (" << _features_dc.size(0) << ", " << _features_dc.size(1) << ", " << _features_dc.size(2) << ")" << std::endl;
    _features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous().set_requires_grad(true);
    std::cout << "_features_rest size after transpose: (" << _features_rest.size(0) << ", " << _features_rest.size(1) << ", " << _features_rest.size(2) << ")" << std::endl;
    _scaling = scales.set_requires_grad(true);
    _rotation = rots.set_requires_grad(true);
    _opacity = opacities.set_requires_grad(true);
    _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA);
    std::cout << "Creating from pcd done" << std::endl;
}

/**
 * @brief Setup the Gaussian Model for training
 *
 * This function sets up the Gaussian model for training by initializing several
 * parameters and settings based on the provided OptimizationParameters object.
 *
 * @param params The OptimizationParameters object providing the settings for training
 */
void GaussianModel::Training_setup(const OptimizationParameters& params) {
    std::cout << "Training setup" << std::endl;
    this->_percent_dense = params.percent_dense;
    this->_xyz_gradient_accum = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);
    this->_denom = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);
    this->_xyz_scheduler_args = Expon_lr_func(params.position_lr_init * this->_spatial_lr_scale,
                                              params.position_lr_final * this->_spatial_lr_scale,
                                              params.position_lr_delay_mult,
                                              params.position_lr_max_steps);

    // TODO: seems kind weird to do this here
    _optimizer = std::make_unique<torch::optim::Adam>(parameters(), torch::optim::AdamOptions(0.0).eps(1e-15));
    std::cout << "Training setup done" << std::endl;
}

void GaussianModel::Update_learning_rate(float iteration) {
    // This is hacky because you cant change in libtorch individual parameter learning rate
    // xyz is added first, since _optimizer->param_groups() return a vector, we assume that xyz stays first
    static_cast<torch::optim::AdamOptions&>(_optimizer->param_groups()[0].options()).set_lr(_xyz_scheduler_args(iteration));
}

void GaussianModel::Save_as_ply(const std::string& filename) {
    throw std::runtime_error("Not implemented");
}

void GaussianModel::Reset_opacity() {
    // Hopefully this is doing the same as the python code
    std::cout << "Resetting opacity" << std::endl;
    // opacitiy activation
    _opacity = inverse_sigmoid(torch::ones_like(Get_opacity() * 0.01));
    auto* adamParams = static_cast<torch::optim::AdamParamState*>(_optimizer->state()["opacity"].get());
    adamParams->exp_avg(torch::zeros_like(_opacity));
    adamParams->exp_avg_sq(torch::zeros_like(_opacity));
    std::cout << "Opacity resetting done!" << std::endl;
}

void prune_optimizer(torch::optim::Adam* optimizer, const torch::Tensor& mask, torch::Tensor& updateTensor, const std::string& name) {
    std::cout << "Prune Optimizer " << name << std::endl;
    auto* adamParamState = static_cast<torch::optim::AdamParamState*>(optimizer->state()[name].get());
    if (adamParamState != nullptr) {
        adamParamState->exp_avg(adamParamState->exp_avg().masked_select(mask));
        adamParamState->exp_avg_sq(adamParamState->exp_avg_sq().masked_select(mask));
    }
    updateTensor = updateTensor.masked_select(mask);
    std::cout << "Prune Optimizer done!" << name << std::endl;
}

void GaussianModel::prune_points(const torch::Tensor& mask) {
    std::cout << "Pruning points" << std::endl;
    prune_optimizer(_optimizer.get(), mask, _xyz, "xyz");
    prune_optimizer(_optimizer.get(), mask, _features_dc, "features_dc");
    prune_optimizer(_optimizer.get(), mask, _features_rest, "features_rest");
    prune_optimizer(_optimizer.get(), mask, _scaling, "scaling");
    prune_optimizer(_optimizer.get(), mask, _rotation, "rotation");
    prune_optimizer(_optimizer.get(), mask, _opacity, "opacity");

    torch::Tensor valid_points_mask = ~mask;
    _xyz_gradient_accum = _xyz_gradient_accum.masked_select(valid_points_mask);
    _denom = _denom.masked_select(valid_points_mask);
    _max_radii2D = _max_radii2D.masked_select(valid_points_mask);

    std::cout << "Pruning points done!" << std::endl;
}

void cat_tensors_to_optimizer(torch::optim::Adam* optimizer,
                              torch::Tensor& updateTensor,
                              const std::string& name,
                              int param_position) {
    std::cout << "Cat tensors to optimizer" << std::endl;
    auto* adamParamState = static_cast<torch::optim::AdamParamState*>(optimizer->state()[name].get());
    if (adamParamState != nullptr) {
        adamParamState->exp_avg(torch::cat({adamParamState->exp_avg(), torch::zeros_like(updateTensor)}, 0));
        adamParamState->exp_avg_sq(torch::cat({adamParamState->exp_avg_sq(), torch::zeros_like(updateTensor)}, 0));
        if (optimizer->param_groups()[param_position].params().size() != 1) {
            std::cout << "Optimizer param groups should only have one parameter" << std::endl;
            std::cout << "Actual Size: " << optimizer->param_groups()[param_position].params().size() << std::endl;
            throw std::runtime_error("Optimizer param groups should only have one parameter");
        }
        optimizer->param_groups()[param_position].params()[0] = torch::cat({optimizer->param_groups()[param_position].params()[0], updateTensor}, 0);
    } else {
        if (optimizer->param_groups()[param_position].params().size() != 1) {
            std::cout << "Optimizer param groups should only have one parameter" << std::endl;
            std::cout << "Actual Size: " << optimizer->param_groups()[param_position].params().size() << std::endl;
            throw std::runtime_error("Optimizer param groups should only have one parameter");
        }
    }
    optimizer->param_groups()[param_position].params()[0].set_requires_grad(true);
    updateTensor = optimizer->param_groups()[param_position].params()[0];
    std::cout << "Cat tensors to optimizer done!" << std::endl;
}

void GaussianModel::densification_postfix(const torch::Tensor& new_xyz,
                                          const torch::Tensor& new_features_dc,
                                          const torch::Tensor& new_features_rest,
                                          const torch::Tensor& new_scaling,
                                          const torch::Tensor& new_rotation,
                                          const torch::Tensor& new_opacity) {
    std::cout << "Densification postfix" << std::endl;
    cat_tensors_to_optimizer(_optimizer.get(), _xyz, "xyz", 0);
    cat_tensors_to_optimizer(_optimizer.get(), _features_dc, "features_dc", 1);
    cat_tensors_to_optimizer(_optimizer.get(), _features_rest, "features_rest", 2);
    cat_tensors_to_optimizer(_optimizer.get(), _scaling, "scaling", 3);
    cat_tensors_to_optimizer(_optimizer.get(), _rotation, "rotation", 4);
    cat_tensors_to_optimizer(_optimizer.get(), _opacity, "opacity", 5);

    _xyz_gradient_accum = torch::zeros({_xyz.size(0), 1}).to(torch::kCUDA);
    _denom = torch::zeros({_xyz.size(0), 1}).to(torch::kCUDA);
    _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA);
    std::cout << "Densification postfix done!" << std::endl;
}

void GaussianModel::densify_and_split(torch::Tensor& grads, float grad_threshold, float scene_extent, int N) {
    int n_init_points = _xyz.size(0);
    // Extract points that satisfy the gradient condition
    torch::Tensor padded_grad = torch::zeros({n_init_points}).to(torch::kCUDA);
    padded_grad.slice(0, 0, grads.size(0)) = grads.squeeze();
    torch::Tensor selected_pts_mask = torch::where(padded_grad >= grad_threshold, torch::ones_like(padded_grad).to(torch::kBool), torch::zeros_like(padded_grad).to(torch::kBool));
    selected_pts_mask = torch::logical_and(selected_pts_mask, std::get<0>(Get_scaling().max(1)) > _percent_dense * scene_extent);

    torch::Tensor stds = Get_scaling().index_select(0, selected_pts_mask.nonzero().squeeze()).repeat({N, 1}, 0);
    torch::Tensor means = torch::zeros({stds.size(0), 3}).to(torch::kCUDA);
    torch::Tensor samples = torch::randn({stds.size(0), stds.size(1)}).to(torch::kCUDA) * stds + means;
    torch::Tensor rots = build_rotation(_rotation.index_select(0, selected_pts_mask.nonzero().squeeze())).repeat({N, 1, 1}, 0);
    torch::Tensor new_xyz = torch::bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + _xyz.index_select(0, selected_pts_mask.nonzero().squeeze()).repeat({N, 1}, 0);

    // scaling inverse activation immediately after the scaling activation
    torch::Tensor new_scaling = torch::log(Get_scaling().index_select(0, selected_pts_mask.nonzero().squeeze()).repeat({N, 1}, 0) / (0.8 * N));
    torch::Tensor new_rotation = _rotation.index_select(0, selected_pts_mask.nonzero().squeeze()).repeat({N, 1}, 0);
    torch::Tensor new_features_dc = _features_dc.index_select(0, selected_pts_mask.nonzero().squeeze()).repeat({N, 1, 1}, 0);
    torch::Tensor new_features_rest = _features_rest.index_select(0, selected_pts_mask.nonzero().squeeze()).repeat({N, 1, 1}, 0);
    torch::Tensor new_opacity = _opacity.index_select(0, selected_pts_mask.nonzero().squeeze()).repeat({N, 1}, 0);

    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity);

    torch::Tensor prune_filter = torch::cat({selected_pts_mask, torch::zeros({N * selected_pts_mask.sum().item<int>()}).to(torch::kCUDA).to(torch::kBool)});
    prune_points(prune_filter);
}

void GaussianModel::densify_and_clone(torch::Tensor& grads, float grad_threshold, float scene_extent) {
    std::cout << "Densify and clone" << std::endl;
    // Extract points that satisfy the gradient condition
    torch::Tensor selected_pts_mask = torch::where(grads.norm(-1) >= grad_threshold, torch::ones_like(grads).to(torch::kBool), torch::zeros_like(grads).to(torch::kBool));
    selected_pts_mask = torch::logical_and(selected_pts_mask, std::get<0>(Get_scaling().max(1)) <= _percent_dense * scene_extent);

    torch::Tensor new_xyz = _xyz.index_select(0, selected_pts_mask.nonzero().squeeze());
    torch::Tensor new_features_dc = _features_dc.index_select(0, selected_pts_mask.nonzero().squeeze());
    torch::Tensor new_features_rest = _features_rest.index_select(0, selected_pts_mask.nonzero().squeeze());
    torch::Tensor new_opacity = _opacity.index_select(0, selected_pts_mask.nonzero().squeeze());
    torch::Tensor new_scaling = _scaling.index_select(0, selected_pts_mask.nonzero().squeeze());
    torch::Tensor new_rotation = _rotation.index_select(0, selected_pts_mask.nonzero().squeeze());

    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity);
    std::cout << "Densify and clone done!" << std::endl;
}

void GaussianModel::Densify_and_prune(float max_grad, float min_opacity, float extent, float max_screen_size) {
    torch::Tensor grads = _xyz_gradient_accum / _denom;
    grads.index_put_({grads.isnan()}, 0.0);

    densify_and_clone(grads, max_grad, extent);
    densify_and_split(grads, max_grad, extent);

    torch::Tensor prune_mask = (Get_opacity() < min_opacity).squeeze();
    if (max_screen_size > 0) {
        torch::Tensor big_points_vs = _max_radii2D > max_screen_size;
        torch::Tensor big_points_ws = std::get<0>(Get_scaling().max(1)) > 0.1 * extent;
        prune_mask = torch::logical_or(prune_mask, torch::logical_or(big_points_vs, big_points_ws));
    }
    prune_points(prune_mask);
}

void GaussianModel::Add_densification_stats(torch::Tensor& viewspace_point_tensor, torch::Tensor& update_filter) {
    _xyz_gradient_accum.index_put_({update_filter}, _xyz_gradient_accum.index_select(0, update_filter.nonzero().squeeze()) + viewspace_point_tensor.grad().index_select(0, update_filter.nonzero().squeeze()).slice(1, 0, 2).norm(2, -1, true));
    _denom.index_put_({update_filter}, _denom.index_select(0, update_filter.nonzero().squeeze()) + 1);
}
