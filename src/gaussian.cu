#include "gaussian.cuh"
#include <exception>

GaussianModel::GaussianModel(int sh_degree) : max_sh_degree(sh_degree),
                                              active_sh_degree(0),
                                              _xyz_scheduler_args(Expon_lr_func(0.0, 1.0)) {

    _xyz = torch::empty({0});
    _features_dc = torch::empty({0});
    _features_rest = torch::empty({0});
    _scaling = torch::empty({0});
    _rotation = torch::empty({0});
    _opacity = torch::empty({0});
    _max_radii2D = torch::empty({0});
    _xyz_gradient_accum = torch::empty({0});
    _optimizer = nullptr;

    // IMPORTANT: The order has to stay fix because we must now the place of the parameters
    // The optimizer just gets the tensor. There is currently no way to access the parameters by name
    register_parameter("xyz", _xyz, true);
    register_parameter("features_dc", _features_dc, true);
    register_parameter("features_rest", _features_rest, true);
    register_parameter("scaling", _scaling, true);
    register_parameter("rotation", _rotation, true);
    register_parameter("opacity", _opacity, true);

    // Scaling activation and its inverse
    _scaling_activation = torch::exp;
    _scaling_inverse_activation = torch::log;

    // Covariance activation function
    _covariance_activation = [](const torch::Tensor& scaling, const torch::Tensor& scaling_modifier, const torch::Tensor& rotation) {
        auto L = build_scaling_rotation(scaling_modifier * scaling, rotation);
        auto actual_covariance = torch::mm(L, L.transpose(1, 2));
        auto symm = strip_symmetric(actual_covariance);
        return symm;
    };

    // Opacity activation and its inverse
    _opacity_activation = torch::sigmoid;
    _inverse_opacity_activation = inverse_sigmoid;

    // Rotation activation function
    _rotation_activation = torch::nn::functional::normalize;
}

/**
 * @brief Fetches the features of the Gaussian model
 *
 * This function concatenates _features_dc and _features_rest along the second dimension.
 *
 * @return Tensor of the concatenated features
 */
torch::Tensor GaussianModel::get_features() const {
    auto features_dc = _features_dc;
    auto features_rest = _features_rest;
    return torch::cat({features_dc, features_rest}, 1);
}

/**
 * @brief Increment the SH degree by 1
 *
 * This function increments the active_sh_degree by 1, up to a maximum of max_sh_degree.
 */
void GaussianModel::OneupSHdegree() {
    if (active_sh_degree < max_sh_degree) {
        active_sh_degree++;
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

    auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((max_sh_degree + 1), 2))}).to(torch::kCUDA);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None), torch::indexing::Slice(1, torch::indexing::None)}, 0.0);

    std::cout << "Number of points at initialisation : " << fused_point_cloud.size(0) << std::endl;

    auto dist2 = torch::clamp_min(distCUDA2(torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size()), 3}, pointType).to(torch::kCUDA)), 0.0000001);

    auto scales = torch::log(torch::sqrt(dist2)).unsqueeze(-1).repeat({1, 3});
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
    this->percent_dense = params.percent_dense;
    this->_xyz_gradient_accum = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);
    this->_denom = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);

    _optimizer = std::make_unique<torch::optim::Adam>(parameters(), torch::optim::AdamOptions(0.0).eps(1e-15));
    this->_xyz_scheduler_args = Expon_lr_func(params.position_lr_init * this->_spatial_lr_scale,
                                              params.position_lr_final * this->_spatial_lr_scale,
                                              params.position_lr_delay_mult,
                                              params.position_lr_max_steps);
}

void GaussianModel::Update_Learning_Rate(float lr) {
    // This is hacky because you cant change in libtorch individual parameter learning rate
    // xyz is added first, since _optimizer->param_groups() return a vector, we assume that xyz stays first
    static_cast<torch::optim::AdamOptions&>(_optimizer->param_groups()[0].options()).set_lr(lr);
}

void GaussianModel::Save_As_PLY(const std::string& filename) {
    throw std::runtime_error("Not implemented");
}

void GaussianModel::Reset_Opacity() {
    // Hopefully this is doing the same as the python code
    std::cout << "Resetting opacity" << std::endl;
    _opacity = inverse_sigmoid(torch::ones_like(get_opacity() * 0.01));
    auto* adamParams = static_cast<torch::optim::AdamParamState*>(_optimizer->state()["opacity"].get());
    adamParams->exp_avg(torch::zeros_like(_opacity));
    adamParams->exp_avg_sq(torch::zeros_like(_opacity));
    std::cout << "Opacity resetting done!" << std::endl;
}

void GaussianModel::prune_optimizer(const torch::Tensor& mask, torch::Tensor& updateTensor, const std::string& name) {
    auto* adamParams = static_cast<torch::optim::AdamParamState*>(_optimizer->state()["opacity"].get());
    if (adamParams != nullptr) {
        adamParams->exp_avg(adamParams->exp_avg().masked_select(mask));
        adamParams->exp_avg_sq(adamParams->exp_avg_sq().masked_select(mask));
    }
    updateTensor = updateTensor.masked_select(mask);
}

void GaussianModel::prune_points(const torch::Tensor& mask) {
    std::cout << "Pruning points" << std::endl;
    prune_optimizer(mask, _xyz, "xyz");
    prune_optimizer(mask, _features_dc, "features_dc");
    prune_optimizer(mask, _features_rest, "features_rest");
    prune_optimizer(mask, _scaling, "scaling");
    prune_optimizer(mask, _rotation, "rotation");
    prune_optimizer(mask, _opacity, "opacity");

    torch::Tensor valid_points_mask = ~mask;
    _xyz_gradient_accum = _xyz_gradient_accum.masked_select(valid_points_mask);
    _denom = _denom.masked_select(valid_points_mask);
    _max_radii2D = _max_radii2D.masked_select(valid_points_mask);
}