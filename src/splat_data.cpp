#include "core/splat_data.hpp"
#include "core/exporter.hpp"
#include "core/mean_neighbor_dist.hpp"
#include "core/scene_info.hpp"
#include <torch/torch.h>

// Constructor from tensors
SplatData::SplatData(int sh_degree,
                     torch::Tensor xyz,
                     torch::Tensor features_dc,
                     torch::Tensor features_rest,
                     torch::Tensor scaling,
                     torch::Tensor rotation,
                     torch::Tensor opacity,
                     float scene_scale)
    : _max_sh_degree{sh_degree},
      _active_sh_degree{0},
      _scene_scale{scene_scale},
      _xyz{std::move(xyz)},
      _features_dc{std::move(features_dc)},
      _features_rest{std::move(features_rest)},
      _scaling{std::move(scaling)},
      _rotation{std::move(rotation)},
      _opacity{std::move(opacity)},
      _max_radii2D{torch::zeros({_xyz.size(0)}).to(torch::kCUDA)} {}

// Computed getters
torch::Tensor SplatData::get_xyz() const {
    return _xyz;
}

torch::Tensor SplatData::get_opacity() const {
    return torch::sigmoid(_opacity);
}

torch::Tensor SplatData::get_rotation() const {
    return torch::nn::functional::normalize(_rotation,
                                            torch::nn::functional::NormalizeFuncOptions().dim(-1));
}

torch::Tensor SplatData::get_scaling() const {
    return torch::exp(_scaling);
}

torch::Tensor SplatData::get_features() const {
    return torch::cat({_features_dc, _features_rest}, 1);
}

// Utility method
void SplatData::increment_sh_degree() {
    if (_active_sh_degree < _max_sh_degree) {
        _active_sh_degree++;
    }
}

// Static factory method (like original gaussian_init)
SplatData SplatData::create_from_point_cloud(PointCloud& pcd, int max_sh_degree, float scene_scale) {
    // Helper lambdas
    auto inverse_sigmoid = [](torch::Tensor x) {
        return torch::log(x / (1 - x));
    };

    auto rgb_to_sh = [](const torch::Tensor& rgb) {
        constexpr float kInvSH = 0.28209479177387814f; // 1 / √(4π)
        return (rgb - 0.5f) / kInvSH;
    };

    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
    const auto f32_cuda = f32.device(torch::kCUDA);

    // 1. xyz
    auto xyz = torch::from_blob(pcd._points.data(),
                                {static_cast<int64_t>(pcd._points.size()), 3},
                                f32)
                   .to(torch::kCUDA)
                   .set_requires_grad(true);

    // 2. scaling (log(σ))
    auto nn_dist = torch::clamp_min(compute_mean_neighbor_distances(xyz), 1e-7);
    auto scaling = torch::log(torch::sqrt(nn_dist))
                       .unsqueeze(-1)
                       .repeat({1, 3})
                       .to(f32_cuda)
                       .set_requires_grad(true);

    // 3. rotation & opacity
    auto rotation = torch::zeros({xyz.size(0), 4}, f32_cuda)
                        .index_put_({torch::indexing::Slice(), 0}, 1)
                        .set_requires_grad(true);

    auto opacity = inverse_sigmoid(0.5f * torch::ones({xyz.size(0), 1}, f32_cuda))
                       .set_requires_grad(true);

    // 4. features (DC + rest)
    auto rgb = torch::from_blob(pcd._colors.data(),
                                {static_cast<int64_t>(pcd._colors.size()), 3},
                                torch::TensorOptions().dtype(torch::kUInt8))
                   .to(f32) /
               255.f;

    auto fused_color = rgb_to_sh(rgb).to(torch::kCUDA);

    const int64_t feature_shape = static_cast<int64_t>(std::pow(max_sh_degree + 1, 2));
    auto features = torch::zeros({fused_color.size(0), 3, feature_shape}, f32_cuda);

    // DC coefficients
    features.index_put_({torch::indexing::Slice(),
                         torch::indexing::Slice(),
                         0},
                        fused_color);

    auto features_dc = features.index({torch::indexing::Slice(),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice(0, 1)})
                           .transpose(1, 2)
                           .contiguous()
                           .set_requires_grad(true);

    auto features_rest = features.index({torch::indexing::Slice(),
                                         torch::indexing::Slice(),
                                         torch::indexing::Slice(1, torch::indexing::None)})
                             .transpose(1, 2)
                             .contiguous()
                             .set_requires_grad(true);

    return SplatData(max_sh_degree, xyz, features_dc, features_rest,
                     scaling, rotation, opacity, scene_scale);
}

GaussianPointCloud SplatData::to_point_cloud() const {
    GaussianPointCloud pc;
    pc.xyz = _xyz.cpu().contiguous();
    pc.normals = torch::zeros_like(pc.xyz);
    pc.features_dc = _features_dc.transpose(1, 2).flatten(1).cpu();
    pc.features_rest = _features_rest.transpose(1, 2).flatten(1).cpu();
    pc.opacity = _opacity.cpu();
    pc.scaling = _scaling.cpu();
    pc.rotation = _rotation.cpu();
    pc.attribute_names = make_attribute_names(_features_dc, _features_rest, _scaling, _rotation);
    return pc;
}