#pragma once

#include <filesystem>
#include <torch/torch.h>

std::tuple<unsigned char*, int, int, int>
load_image(std::filesystem::path p, int res_div = -1);
void save_image(const std::filesystem::path& path, torch::Tensor image);
void save_image(const std::filesystem::path& path,
                const std::vector<torch::Tensor>& images,
                bool horizontal = true,
                int separator_width = 2);

void free_image(unsigned char* image);