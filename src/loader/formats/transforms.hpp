#pragma once
#include "core/camera.hpp"
#include "colmap.hpp"
#include "core/point_cloud.hpp"

#include <filesystem>
#include <vector>

std::tuple<std::vector<CameraData>, torch::Tensor> read_transforms_cameras_and_images(
    const std::filesystem::path& transPath);

gs::PointCloud generate_random_point_cloud();