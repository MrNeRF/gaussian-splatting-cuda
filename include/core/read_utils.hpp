#pragma once
#include "core/camera_info.hpp"
#include "core/point_cloud.hpp"
#include <filesystem>
#include <vector>

// Read COLMAP cameras, images, and compute nerf norm
std::tuple<std::vector<CameraInfo>, float> read_colmap_cameras_and_images(
    const std::filesystem::path& filepath);

// Read COLMAP point cloud
PointCloud read_colmap_point_cloud(const std::filesystem::path& filepath);