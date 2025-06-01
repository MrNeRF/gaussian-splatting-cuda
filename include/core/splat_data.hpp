#pragma once

#include <filesystem>
#include <string>
#include <torch/torch.h>
#include <vector>

// Forward declarations
struct PointCloud;
struct GaussianPointCloud;

class SplatData {
public:
    SplatData() = default;

    // Constructor
    SplatData(int sh_degree,
              torch::Tensor xyz,
              torch::Tensor features_dc,
              torch::Tensor features_rest,
              torch::Tensor scaling,
              torch::Tensor rotation,
              torch::Tensor opacity,
              float scene_scale);

    // Static factory method to create from PointCloud
    static SplatData create_from_point_cloud(PointCloud& pcd, int max_sh_degree, float scene_scale);

    // Computed getters (implemented in cpp)
    torch::Tensor get_xyz() const;
    torch::Tensor get_opacity() const;
    torch::Tensor get_rotation() const;
    torch::Tensor get_scaling() const;
    torch::Tensor get_features() const;

    // Simple inline getters
    inline int get_active_sh_degree() const { return _active_sh_degree; }
    inline float get_scene_scale() const { return _scene_scale; }
    inline int64_t size() const { return _xyz.size(0); }

    // Raw tensor access for optimization (inline for performance)
    inline torch::Tensor& xyz() { return _xyz; }
    inline torch::Tensor& opacity_raw() { return _opacity; }
    inline torch::Tensor& rotation_raw() { return _rotation; }
    inline torch::Tensor& scaling_raw() { return _scaling; }
    inline torch::Tensor& features_dc() { return _features_dc; }
    inline torch::Tensor& features_rest() { return _features_rest; }
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

    torch::Tensor _xyz;
    torch::Tensor _features_dc;
    torch::Tensor _features_rest;
    torch::Tensor _scaling;
    torch::Tensor _rotation;
    torch::Tensor _opacity;
    torch::Tensor _max_radii2D;

    // Convert to point cloud for export (now private)
    GaussianPointCloud to_point_cloud() const;
};