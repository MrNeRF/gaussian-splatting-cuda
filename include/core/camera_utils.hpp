#pragma once

#include <filesystem>

std::tuple<unsigned char*, int, int, int>
load_image(std::filesystem::path p, int res_div = -1);

void free_image(unsigned char* image);
