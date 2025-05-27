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
//  World → view (NeRF++ translate/scale variant)
// -----------------------------------------------------------------------------
torch::Tensor getWorld2View2(const torch::Tensor& R,
                             const torch::Tensor& t,
                             const torch::Tensor& translate,
                             float scale) {
    assert_mat(R, 3, 3, "R");
    assert_vec(t, 3, "t");
    assert_vec(translate, 3, "translate");

    auto dev = R.device();

    // 4×4 homogeneous matrix (row-major in torch)
    torch::Tensor Rt = torch::eye(4,
                                  torch::TensorOptions().dtype(torch::kFloat32).device(dev));

    // ---------------------------  THE ONLY CHANGE  --------------------------
    Rt.index_put_({torch::indexing::Slice(0, 3),
                   torch::indexing::Slice(0, 3)},
                  R.t()); // <── transpose!
    // -----------------------------------------------------------------------

    Rt.index_put_({torch::indexing::Slice(0, 3), 3}, t);

    torch::Tensor C2W = torch::linalg_inv(Rt); // camera → world
    C2W.index_put_({torch::indexing::Slice(0, 3), 3},
                   (C2W.index({torch::indexing::Slice(0, 3), 3}) + translate) * scale);

    return torch::linalg_inv(C2W).clone(); // world → camera
}

// -----------------------------------------------------------------------------
//  Projection matrix (OpenGL style, z-forward)
// -----------------------------------------------------------------------------
torch::Tensor getProjectionMatrix(float znear, float zfar,
                                  float fovX, float fovY) {
    float tanX = std::tan(fovX * 0.5f);
    float tanY = std::tan(fovY * 0.5f);

    float right = tanX * znear;
    float left = -right;
    float top = tanY * znear;
    float bottom = -top;

    float z_sign = 1.0f; // OpenGL right-handed

    torch::Tensor P = torch::zeros({4, 4}, torch::kFloat32);
    P[0][0] = 2 * znear / (right - left);
    P[1][1] = 2 * znear / (top - bottom);
    P[0][2] = (right + left) / (right - left);
    P[1][2] = (top + bottom) / (top - bottom);
    P[3][2] = z_sign;
    P[2][2] = z_sign * zfar / (zfar - znear);
    P[2][3] = -(zfar * znear) / (zfar - znear);
    return P.clone();
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
