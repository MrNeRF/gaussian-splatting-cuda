#pragma once
#include "point_cloud.cuh"
#include <filesystem>

PointCloud read_ply_file(std::filesystem::path filepath);

void read_colmap_scene_info(std::filesystem::path filepath);