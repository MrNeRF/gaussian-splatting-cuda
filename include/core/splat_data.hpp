#pragma once

#include "core/exporter.hpp"
#include "core/gaussian_init.hpp"
#include <torch/torch.h>

class SplatData {
public:
    SplatData() = default;

    SplatData(int sh_degree, gauss::init::InitTensors&& init)
        : _max_sh_degree{sh_degree},
          _active_sh_degree{0},
          _scene_scale{std::move(init.scene_scale)},
          _xyz{std::move(init.xyz)},
          _scaling{std::move(init.scaling)},
          _rotation{std::move(init.rotation)},
          _opacity{std::move(init.opacity)},
          _features_dc{std::move(init.features_dc)},
          _features_rest{std::move(init.features_rest)},
          _max_radii2D{torch::zeros({_xyz.size(0)}).to(torch::kCUDA, /*copy=*/true)} {}

    // Getters
    torch::Tensor get_xyz() const { return _xyz; }
    torch::Tensor get_opacity() const { return torch::sigmoid(_opacity); }
    torch::Tensor get_rotation() const { return torch::nn::functional::normalize(_rotation); }
    torch::Tensor get_scaling() const { return torch::exp(_scaling); }
    torch::Tensor get_features() const {
        return torch::cat({_features_dc, _features_rest}, 1);
    }
    int get_active_sh_degree() const { return _active_sh_degree; }
    float get_scene_scale() const { return _scene_scale; }

    // Raw tensor access for optimization
    torch::Tensor& xyz() { return _xyz; }
    torch::Tensor& opacity_raw() { return _opacity; }
    torch::Tensor& rotation_raw() { return _rotation; }
    torch::Tensor& scaling_raw() { return _scaling; }
    torch::Tensor& features_dc() { return _features_dc; }
    torch::Tensor& features_rest() { return _features_rest; }
    torch::Tensor& max_radii2D() { return _max_radii2D; }

    // Utility methods
    void increment_sh_degree() {
        if (_active_sh_degree < _max_sh_degree) {
            _active_sh_degree++;
        }
    }

    // Convert to point cloud for export
    GaussianPointCloud to_point_cloud() const {
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

    // Get number of points
    int64_t size() const { return _xyz.size(0); }

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
};