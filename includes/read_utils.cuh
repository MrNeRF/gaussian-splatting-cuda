#pragma once
#include "point_cloud.cuh"
#include "scene_info.cuh"
#include <filesystem>
#include <memory>

PointCloud read_ply_file(std::filesystem::path filepath);

std::unique_ptr<SceneInfo> read_colmap_scene_info(std::filesystem::path filepath);