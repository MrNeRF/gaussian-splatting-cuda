#pragma once
#include "core/point_cloud.hpp"
#include "core/scene_info.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

std::unique_ptr<SceneInfo> read_colmap_scene_info(const std::filesystem::path& filepath, int resolution);
