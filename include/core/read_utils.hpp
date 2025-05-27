#pragma once
#include "core/point_cloud.hpp"
#include "core/scene_info.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

PointCloud read_ply_file(std::filesystem::path filepath);
std::unique_ptr<SceneInfo> read_colmap_scene_info(std::filesystem::path filepath, int resolution);
void Write_output_ply(const std::filesystem::path& file_path, const std::vector<torch::Tensor>& tensors, const std::vector<std::string>& attributes_names);
