#pragma once
#include "core/scene_info.hpp"
#include <filesystem>
#include <memory>

std::unique_ptr<SceneInfo> read_colmap_scene_info(const std::filesystem::path& filepath);
