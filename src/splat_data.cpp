#include "core/splat_data.hpp"
#include "core/mean_neighbor_dist.hpp"
#include "core/scene_info.hpp"
#include "external/tinyply.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

struct GaussianPointCloud {
    torch::Tensor xyz, normals,
        features_dc, features_rest,
        opacity, scaling, rotation;
    std::vector<std::string> attribute_names;
};

static inline void write_ply(const GaussianPointCloud& pc,
                             const std::filesystem::path& root,
                             int iteration,
                             bool join_thread = false) {
    namespace fs = std::filesystem;
    fs::path folder = root / ("point_cloud/iteration_" + std::to_string(iteration));
    fs::create_directories(folder);

    /* ----- pack all per-vertex tensors in the order we want to write ----- */
    std::vector<torch::Tensor> tensors{
        pc.xyz, pc.normals, pc.features_dc, pc.features_rest,
        pc.opacity, pc.scaling, pc.rotation};

    /* ----- background job ------------------------------------------------ */
    std::thread t([folder,
                   tensors = std::move(tensors),
                   names = pc.attribute_names]() mutable {
        /* ---- local lambda that owns the actual tinyply call ------------- */
        auto write_output_ply =
            [](const fs::path& file_path,
               const std::vector<torch::Tensor>& data,
               const std::vector<std::string>& attr_names) {
                tinyply::PlyFile ply;
                size_t attr_off = 0;

                for (const auto& tensor : data) {
                    const size_t cols = tensor.size(1);
                    std::vector<std::string> attrs(attr_names.begin() + attr_off,
                                                   attr_names.begin() + attr_off + cols);

                    ply.add_properties_to_element(
                        "vertex",
                        attrs,
                        tinyply::Type::FLOAT32,
                        tensor.size(0),
                        reinterpret_cast<uint8_t*>(tensor.data_ptr<float>()),
                        tinyply::Type::INVALID, 0);

                    attr_off += cols;
                }

                std::filebuf fb;
                fb.open(file_path, std::ios::out | std::ios::binary);
                std::ostream out_stream(&fb);
                ply.write(out_stream, /*binary=*/true);
            };

        write_output_ply(folder / "point_cloud.ply", tensors, names);
    });

    join_thread ? t.join()
                : t.detach();
}

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

// Get attribute names for PLY format
std::vector<std::string> SplatData::get_attribute_names() const {
    std::vector<std::string> a{"x", "y", "z", "nx", "ny", "nz"};

    for (int i = 0; i < _features_dc.size(1) * _features_dc.size(2); ++i)
        a.emplace_back("f_dc_" + std::to_string(i));
    for (int i = 0; i < _features_rest.size(1) * _features_rest.size(2); ++i)
        a.emplace_back("f_rest_" + std::to_string(i));

    a.emplace_back("opacity");

    for (int i = 0; i < _scaling.size(1); ++i)
        a.emplace_back("scale_" + std::to_string(i));
    for (int i = 0; i < _rotation.size(1); ++i)
        a.emplace_back("rot_" + std::to_string(i));

    return a;
}

// Export to PLY
void SplatData::save_ply(const std::filesystem::path& root, int iteration, bool join_thread) const {
    auto pc = to_point_cloud();
    write_ply(pc, root, iteration, join_thread);
}

// Convert to point cloud (now private)
GaussianPointCloud SplatData::to_point_cloud() const {
    GaussianPointCloud pc;
    pc.xyz = _xyz.cpu().contiguous();
    pc.normals = torch::zeros_like(pc.xyz);
    pc.features_dc = _features_dc.transpose(1, 2).flatten(1).cpu();
    pc.features_rest = _features_rest.transpose(1, 2).flatten(1).cpu();
    pc.opacity = _opacity.cpu();
    pc.scaling = _scaling.cpu();
    pc.rotation = _rotation.cpu();
    pc.attribute_names = get_attribute_names();
    return pc;
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