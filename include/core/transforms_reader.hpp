#pragma once
#include "core/camera.hpp"
#include "core/point_cloud.hpp"
#include "core/colmap_reader.hpp"
#include "Common.h"

#include <filesystem>
#include <vector>



std::tuple<std::vector<CameraData>, torch::Tensor> read_transforms_cameras_and_images(
    const std::filesystem::path& base,
    const std::string& images_folder);

PointCloud generate_random_point_cloud();