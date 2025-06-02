#include "core/splat_data.hpp"
#include "core/mean_neighbor_dist.hpp"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "core/read_utils.hpp"
#include "external/tinyply.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

static inline void write_ply(const PointCloud& pc,
                             const std::filesystem::path& root,
                             int iteration,
                             bool join_thread = false) {
    namespace fs = std::filesystem;
    fs::path folder = root / ("point_cloud/iteration_" + std::to_string(iteration));
    fs::create_directories(folder);

    /* ----- pack all per-vertex tensors in the order we want to write ----- */
    std::vector<torch::Tensor> tensors;

    // Always include positions
    tensors.push_back(pc.positions);

    // Only include other tensors if they're defined (for Gaussian point clouds)
    if (pc.normals.defined())
        tensors.push_back(pc.normals);
    if (pc.features_dc.defined())
        tensors.push_back(pc.features_dc);
    if (pc.features_rest.defined())
        tensors.push_back(pc.features_rest);
    if (pc.opacity.defined())
        tensors.push_back(pc.opacity);
    if (pc.scaling.defined())
        tensors.push_back(pc.scaling);
    if (pc.rotation.defined())
        tensors.push_back(pc.rotation);

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

    join_thread ? t.join() : t.detach();
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

PointCloud SplatData::to_point_cloud() const {
    PointCloud pc;

    // Basic attributes
    pc.positions = _xyz.cpu().contiguous();
    pc.normals = torch::zeros_like(pc.positions);

    // Gaussian attributes
    pc.features_dc = _features_dc.transpose(1, 2).flatten(1).cpu();
    pc.features_rest = _features_rest.transpose(1, 2).flatten(1).cpu();
    pc.opacity = _opacity.cpu();
    pc.scaling = _scaling.cpu();
    pc.rotation = _rotation.cpu();

    // Set attribute names for PLY export
    pc.attribute_names = get_attribute_names();

    return pc;
}

SplatData SplatData::init_model_from_pointcloud(const gs::param::TrainingParameters& params, float scene_scale) {
    // Helper lambdas

    auto pcd = read_colmap_point_cloud(params.dataset.data_path);

    auto inverse_sigmoid = [](torch::Tensor x) {
        return torch::log(x / (1 - x));
    };

    auto rgb_to_sh = [](const torch::Tensor& rgb) {
        constexpr float kInvSH = 0.28209479177387814f; // 1 / √(4π)
        return (rgb - 0.5f) / kInvSH;
    };

    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
    const auto f32_cuda = f32.device(torch::kCUDA);

    // Ensure colors are normalized floats
    pcd.normalize_colors();

    // 1. xyz - already a tensor, just move to CUDA and set requires_grad
    auto xyz = pcd.positions.to(torch::kCUDA).set_requires_grad(true);

    // 2. scaling (log(σ)) - compute nearest neighbor distances
    auto nn_dist = torch::clamp_min(compute_mean_neighbor_distances(xyz), 1e-7);
    auto scaling = torch::log(torch::sqrt(nn_dist))
                       .unsqueeze(-1)
                       .repeat({1, 3})
                       .to(f32_cuda)
                       .set_requires_grad(true);

    // 3. rotation (quaternion, identity) - split into multiple lines to avoid compilation error
    auto rotation = torch::zeros({xyz.size(0), 4}, f32_cuda);
    rotation.index_put_({torch::indexing::Slice(), 0}, 1);
    rotation = rotation.set_requires_grad(true);

    // 4. opacity (inverse sigmoid of 0.5)
    auto opacity = inverse_sigmoid(0.5f * torch::ones({xyz.size(0), 1}, f32_cuda))
                       .set_requires_grad(true);

    // 5. features (SH coefficients)
    // Colors are already normalized to float by pcd.normalize_colors()
    auto colors_float = pcd.colors.to(torch::kCUDA);
    auto fused_color = rgb_to_sh(colors_float);

    const int64_t feature_shape = static_cast<int64_t>(std::pow(params.optimization.sh_degree + 1, 2));
    auto features = torch::zeros({fused_color.size(0), 3, feature_shape}, f32_cuda);

    // Set DC coefficients
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

    return SplatData(params.optimization.sh_degree, xyz, features_dc, features_rest, scaling, rotation, opacity, scene_scale);
}