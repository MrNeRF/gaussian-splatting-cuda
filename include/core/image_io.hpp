#pragma once

#include <filesystem>
#include <future>
#include <torch/torch.h>

std::tuple<unsigned char*, int, int, int>
load_image(std::filesystem::path p, int res_div = -1);
void save_image(const std::filesystem::path& path, torch::Tensor image);

void free_image(unsigned char* image);

// Async image loading
std::future<std::tuple<unsigned char*, int, int, int>>
load_image_async(const std::filesystem::path& p, int res_div = -1);