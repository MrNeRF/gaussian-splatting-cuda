#pragma once
#include "core/camera.hpp"

#include <filesystem>
#include <torch/torch.h>
#include <tuple>

// ---------------------------------------------------------------------------
//  World-to-view (NeRF++ variant: optional translate + uniform scale)
//    R : (3×3)  rotation
//    t : (3)    translation
//    translate : (3) shift applied to camera centre in world space
// ---------------------------------------------------------------------------
torch::Tensor getWorld2View2(const torch::Tensor& R,
                             const torch::Tensor& t,
                             const torch::Tensor& translate = torch::Tensor(),
                             float scale = 1.0f);

// Classic OpenGL-style perspective matrix (z-forward, right-handed)
torch::Tensor getProjectionMatrix(float znear, float zfar,
                                  float fovX, float fovY);

inline float fov2focal(float fov_rad, int pixels) {
    return pixels / (2.0f * std::tan(fov_rad * 0.5f));
}

inline float focal2fov(float focal, int pixels) {
    return 2.0f * std::atan(pixels / (2.0f * focal));
}

// Simple stb wrappers (optionally down-sample by 2/4/8)
std::tuple<unsigned char*, int, int, int>
read_image(std::filesystem::path image_path, int resolution = -1);

void free_image(unsigned char* image);
