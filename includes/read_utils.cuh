#pragma once
#include <filesystem>

void read_ply_file(std::filesystem::path filepath);

void read_colmap_scene_info(std::filesystem::path filepath);