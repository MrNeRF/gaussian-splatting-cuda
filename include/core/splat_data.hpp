// Updated splat_data.hpp
#pragma once

#include "core/point_cloud.hpp"
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

namespace gs::param {
    struct TrainingParameters;
}

class SplatData {
public:
    SplatData() = default;
    ~SplatData();

    SplatData(const SplatData&) = delete;
    SplatData& operator=(const SplatData&) = delete;
    SplatData(SplatData&& other) noexcept;
    SplatData& operator=(SplatData&& other) noexcept;

    // Constructor
    SplatData(int sh_degree,
              torch::Tensor means,
              torch::Tensor sh0,
              torch::Tensor shN,
              torch::Tensor scaling,
              torch::Tensor rotation,
              torch::Tensor opacity,
              float scene_scale);

    // Static factory method to create from PointCloud
    static SplatData init_model_from_pointcloud(const gs::param::TrainingParameters& params, torch::Tensor scene_center);

    // Computed getters (implemented in cpp)
    torch::Tensor get_means() const;
    torch::Tensor get_opacity() const;
    torch::Tensor get_rotation() const;
    torch::Tensor get_scaling() const;
    torch::Tensor get_shs() const;

    // Simple inline getters
    inline int get_active_sh_degree() const { return _active_sh_degree; }
    inline float get_scene_scale() const { return _scene_scale; }
    inline int64_t size() const { return _means.size(0); }

    // Raw tensor access for optimization (inline for performance)
    inline torch::Tensor& means() { return _means; }
    inline torch::Tensor& opacity_raw() { return _opacity; }
    inline torch::Tensor& rotation_raw() { return _rotation; }
    inline torch::Tensor& scaling_raw() { return _scaling; }
    inline torch::Tensor& sh0() { return _sh0; }
    inline torch::Tensor& shN() { return _shN; }
    inline torch::Tensor& max_radii2D() { return _max_radii2D; }

    // Utility methods
    void increment_sh_degree();

    // Export methods - clean public interface
    void save_ply(const std::filesystem::path& root, int iteration, bool join_thread = false) const;

    // Get attribute names for the PLY format
    std::vector<std::string> get_attribute_names() const;

private:
    int _active_sh_degree = 0;
    int _max_sh_degree = 0;
    float _scene_scale = 0.f;

    torch::Tensor _means;
    torch::Tensor _sh0;
    torch::Tensor _shN;
    torch::Tensor _scaling;
    torch::Tensor _rotation;
    torch::Tensor _opacity;
    torch::Tensor _max_radii2D;

    // Thread management for async saves
    mutable std::vector<std::thread> _save_threads;
    mutable std::mutex _threads_mutex;

    // Convert to point cloud for export
    PointCloud to_point_cloud() const;

    // Helper to clean up finished threads
    void cleanup_finished_threads() const;
};