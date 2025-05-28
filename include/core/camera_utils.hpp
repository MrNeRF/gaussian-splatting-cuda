#pragma once

#include <filesystem>

std::tuple<unsigned char*, int, int, int>
read_image(std::filesystem::path image_path, int resolution = -1);

void free_image(unsigned char* image);
