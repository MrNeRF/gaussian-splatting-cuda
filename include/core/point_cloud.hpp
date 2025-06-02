#pragma once

#include <string>
#include <torch/torch.h>
#include <vector>

// Unified point cloud structure using torch tensors from the start
struct PointCloud {
    torch::Tensor means;  // [N, 3] float32
    torch::Tensor colors; // [N, 3] uint8 or float32

    // For Gaussian point clouds (optional, can be empty for basic point clouds)
    torch::Tensor normals;  // [N, 3] float32
    torch::Tensor sh0;      // [N, 3, 1] float32
    torch::Tensor shN;      // [N, 3, (sh_degree+1)^2-1] float32
    torch::Tensor opacity;  // [N, 1] float32
    torch::Tensor scaling;  // [N, 3] float32
    torch::Tensor rotation; // [N, 4] float32

    // Metadata
    std::vector<std::string> attribute_names;

    // Constructor for basic point cloud (means + colors only)
    PointCloud(torch::Tensor pos, torch::Tensor col)
        : means(std::move(pos)),
          colors(std::move(col)) {}

    // Default constructor
    PointCloud() = default;

    // Check if this is a Gaussian point cloud (has additional attributes)
    bool is_gaussian() const {
        return sh0.defined() && sh0.numel() > 0;
    }

    // Get number of points
    int64_t size() const {
        return means.defined() ? means.size(0) : 0;
    }

    // Move to device
    PointCloud to(torch::Device device) const {
        PointCloud pc;
        pc.means = means.defined() ? means.to(device) : means;
        pc.colors = colors.defined() ? colors.to(device) : colors;
        pc.normals = normals.defined() ? normals.to(device) : normals;
        pc.sh0 = sh0.defined() ? sh0.to(device) : sh0;
        pc.shN = shN.defined() ? shN.to(device) : shN;
        pc.opacity = opacity.defined() ? opacity.to(device) : opacity;
        pc.scaling = scaling.defined() ? scaling.to(device) : scaling;
        pc.rotation = rotation.defined() ? rotation.to(device) : rotation;
        pc.attribute_names = attribute_names;
        return pc;
    }

    // Convert colors to float [0,1] if they're uint8
    void normalize_colors() {
        if (colors.defined() && colors.dtype() == torch::kUInt8) {
            colors = colors.to(torch::kFloat32) / 255.0f;
        }
    }
};