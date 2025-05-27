#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "core/camera_utils.hpp"
#include "core/torch_shapes.hpp"
#include "external/stb_image.h"
#include "external/stb_image_resize.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <torch/torch.h>

// -----------------------------------------------------------------------------
//  World → view (NeRF++ translate / scale variant)
// -----------------------------------------------------------------------------
torch::Tensor getWorld2View2(const torch::Tensor& R,
                             const torch::Tensor& t)
{
    assert_mat(R, 3, 3, "R");
    assert_vec(t, 3, "t");

    const auto dev = R.device();
    torch::Tensor Rt = torch::eye(4,
                                  torch::TensorOptions().dtype(torch::kFloat32).device(dev));

    // 1. Put R^T in the top-left block
    Rt.index_put_({torch::indexing::Slice(0,3),
                   torch::indexing::Slice(0,3)},
                  R.t());

    // 2. Copy translation exactly
    Rt.index_put_({3, torch::indexing::Slice(0,3)}, t);

    // 3. No final transpose needed
    return Rt;
}

// -----------------------------------------------------------------------------
//  Projection matrix (OpenGL style, z-forward)
// -----------------------------------------------------------------------------
torch::Tensor getProjectionMatrix(float znear, float zfar, float fovX, float fovY) {
    float tanHalfFovY = std::tan((fovY / 2.f));
    float tanHalfFovX = std::tan((fovX / 2.f));

    float top = tanHalfFovY * znear;
    float bottom = -top;
    float right = tanHalfFovX * znear;
    float left = -right;

    torch::Tensor P = torch::zeros({4,4}, torch::kFloat32);

    float z_sign = 1.f;

    P[0, 0] = 2.f * znear / (right - left);
    P[1, 1] = 2.f * znear / (top - bottom);
    P[0, 2] = (right + left) / (right - left);
    P[1, 2] = (top + bottom) / (top - bottom);
    P[3, 2] = z_sign;
    P[2, 2] = z_sign * zfar / (zfar - znear);
    P[2, 3] = -(zfar * znear) / (zfar - znear);

    // clone the tensor to allocate new memory
    return P.t().clone();
}


// -----------------------------------------------------------------------------
//  Image I/O helpers
// -----------------------------------------------------------------------------
std::tuple<unsigned char*, int, int, int>
read_image(std::filesystem::path p, int res_div) {
    int w, h, c;
    unsigned char* img = stbi_load(p.string().c_str(), &w, &h, &c, 0);
    if (!img)
        throw std::runtime_error("Load failed: " + p.string() + " : " + stbi_failure_reason());

    if (res_div == 2 || res_div == 4 || res_div == 8) {
        int nw = w / res_div, nh = h / res_div;
        auto* out = static_cast<unsigned char*>(malloc(nw * nh * c));
        if (!stbir_resize_uint8(img, w, h, 0, out, nw, nh, 0, c))
            throw std::runtime_error("Resize failed: " + p.string() + " : " + stbi_failure_reason());
        stbi_image_free(img);
        img = out;
        w = nw;
        h = nh;
    }
    return {img, w, h, c};
}

void free_image(unsigned char* img) { stbi_image_free(img); }
