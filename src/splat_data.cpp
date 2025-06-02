#include "core/splat_data.hpp"
#include "core/colmap_reader.hpp"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "external/nanoflann.hpp"
#include "external/tinyply.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

namespace {
    // Point cloud adaptor for nanoflann
    struct PointCloudAdaptor {
        const float* points;
        size_t num_points;

        PointCloudAdaptor(const float* pts, size_t n) : points(pts),
                                                        num_points(n) {}

        inline size_t kdtree_get_point_count() const { return num_points; }

        inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
            return points[idx * 3 + dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
    };

    // Fixed: KDTree typedef on single line
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, PointCloudAdaptor, 3>;

    // Compute mean distance to 3 nearest neighbors for each point
    torch::Tensor compute_mean_neighbor_distances(const torch::Tensor& points) {
        auto cpu_points = points.to(torch::kCPU).contiguous();
        const int num_points = cpu_points.size(0);

        TORCH_CHECK(cpu_points.dim() == 2 && cpu_points.size(1) == 3,
                    "Input points must have shape [N, 3]");
        TORCH_CHECK(cpu_points.dtype() == torch::kFloat32,
                    "Input points must be float32");

        if (num_points <= 1) {
            return torch::full({num_points}, 0.01f, points.options());
        }

        const float* data = cpu_points.data_ptr<float>();

        PointCloudAdaptor cloud(data, num_points);
        KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        index.buildIndex();

        auto result = torch::zeros({num_points}, torch::kFloat32);
        float* result_data = result.data_ptr<float>();

#pragma omp parallel for if (num_points > 1000)
        for (int i = 0; i < num_points; i++) {
            const float query_pt[3] = {data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]};

            const size_t num_results = std::min(4, num_points);
            std::vector<size_t> ret_indices(num_results);
            std::vector<float> out_dists_sqr(num_results);

            nanoflann::KNNResultSet<float> resultSet(num_results);
            resultSet.init(&ret_indices[0], &out_dists_sqr[0]);
            index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParameters(10));

            float sum_dist = 0.0f;
            int valid_neighbors = 0;

            for (size_t j = 0; j < num_results && valid_neighbors < 3; j++) {
                if (out_dists_sqr[j] > 1e-8f) {
                    sum_dist += std::sqrt(out_dists_sqr[j]);
                    valid_neighbors++;
                }
            }

            result_data[i] = (valid_neighbors > 0) ? (sum_dist / valid_neighbors) : 0.01f;
        }

        return result.to(points.device());
    }

    void write_ply(const PointCloud& pc,
                   const std::filesystem::path& root,
                   int iteration,
                   bool join_thread = false) {
        namespace fs = std::filesystem;
        fs::path folder = root / ("point_cloud/iteration_" + std::to_string(iteration));
        fs::create_directories(folder);

        std::vector<torch::Tensor> tensors;
        tensors.push_back(pc.means);

        if (pc.normals.defined())
            tensors.push_back(pc.normals);
        if (pc.sh0.defined())
            tensors.push_back(pc.sh0);
        if (pc.shN.defined())
            tensors.push_back(pc.shN);
        if (pc.opacity.defined())
            tensors.push_back(pc.opacity);
        if (pc.scaling.defined())
            tensors.push_back(pc.scaling);
        if (pc.rotation.defined())
            tensors.push_back(pc.rotation);

        std::thread t([folder,
                       tensors = std::move(tensors),
                       names = pc.attribute_names]() mutable {
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
} // namespace

// Constructor from tensors
SplatData::SplatData(int sh_degree,
                     torch::Tensor xyz,
                     torch::Tensor sh0,
                     torch::Tensor shN,
                     torch::Tensor scaling,
                     torch::Tensor rotation,
                     torch::Tensor opacity,
                     float scene_scale)
    : _max_sh_degree{sh_degree},
      _active_sh_degree{0},
      _scene_scale{scene_scale},
      _xyz{std::move(xyz)},
      _sh0{std::move(sh0)},
      _shN{std::move(shN)},
      _scaling{std::move(scaling)},
      _rotation{std::move(rotation)},
      _opacity{std::move(opacity)},
      _max_radii2D{torch::zeros({_xyz.size(0)}).to(torch::kCUDA)} {}

// Computed getters
torch::Tensor SplatData::get_xyz() const {
    return _xyz;
}

torch::Tensor SplatData::get_opacity() const {
    return torch::sigmoid(_opacity).squeeze(-1);
}

torch::Tensor SplatData::get_rotation() const {
    return torch::nn::functional::normalize(_rotation,
                                            torch::nn::functional::NormalizeFuncOptions().dim(-1));
}

torch::Tensor SplatData::get_scaling() const {
    return torch::exp(_scaling);
}

torch::Tensor SplatData::get_features() const {
    return torch::cat({_sh0, _shN}, 1);
}

// Utility method
void SplatData::increment_sh_degree() {
    if (_active_sh_degree < _max_sh_degree) {
        _active_sh_degree++;
        std::cout << "SH degree incremented to: " << _active_sh_degree
                  << " (using " << (_active_sh_degree + 1) * (_active_sh_degree + 1)
                  << " coefficients)" << std::endl;
    }
}

// Get attribute names for PLY format
std::vector<std::string> SplatData::get_attribute_names() const {
    std::vector<std::string> a{"x", "y", "z", "nx", "ny", "nz"};

    for (int i = 0; i < _sh0.size(1) * _sh0.size(2); ++i)
        a.emplace_back("f_dc_" + std::to_string(i));
    for (int i = 0; i < _shN.size(1) * _shN.size(2); ++i)
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
    pc.means = _xyz.cpu().contiguous();
    pc.normals = torch::zeros_like(pc.means);

    // Gaussian attributes
    pc.sh0 = _sh0.transpose(1, 2).flatten(1).cpu();
    pc.shN = _shN.transpose(1, 2).flatten(1).cpu();
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
    auto xyz = pcd.means.to(torch::kCUDA).set_requires_grad(true);

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

    auto sh0 = features.index({torch::indexing::Slice(),
                               torch::indexing::Slice(),
                               torch::indexing::Slice(0, 1)})
                   .transpose(1, 2)
                   .contiguous()
                   .set_requires_grad(true);

    auto shN = features.index({torch::indexing::Slice(),
                               torch::indexing::Slice(),
                               torch::indexing::Slice(1, torch::indexing::None)})
                   .transpose(1, 2)
                   .contiguous()
                   .set_requires_grad(true);

    std::cout << "Initialized SplatData with:" << std::endl;
    std::cout << "  - " << xyz.size(0) << " points" << std::endl;
    std::cout << "  - Max SH degree: " << params.optimization.sh_degree << std::endl;
    std::cout << "  - Total SH coefficients: " << feature_shape << std::endl;
    std::cout << "  - sh0 shape: " << sh0.sizes() << std::endl;
    std::cout << "  - shN shape: " << shN.sizes() << std::endl;

    return SplatData(params.optimization.sh_degree, xyz, sh0, shN, scaling, rotation, opacity, scene_scale);
}