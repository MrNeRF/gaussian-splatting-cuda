#pragma once

#include "core/torch_shapes.hpp"
#include <string>
#include <torch/torch.h>

namespace F = torch::nn::functional;
// -----------------------------------------------------------------------------
//  Image
// -----------------------------------------------------------------------------
class Image {
public:
    Image() = default;
    explicit Image(uint32_t id)
        : _image_ID(id) {}

    uint32_t GetImageID() const noexcept { return _image_ID; }

    // --- setters with shape checks ------------------------------------------
    void set_qvec(const torch::Tensor& q) {
        assert_vec(q, 4, "qvec");
        _qvec = torch::nn::functional::normalize(
            q.to(torch::kFloat32),
            torch::nn::functional::NormalizeFuncOptions().dim(0));
    }
    void set_tvec(const torch::Tensor& t) {
        assert_vec(t, 3, "tvec");
        _tvec = t.to(torch::kFloat32).clone();
    }

    // --- public data members (POD style, accessed directly by parsers) ------
    uint32_t _camera_id = 0;
    std::string _name;

    // Unit quaternion (w,x,y,z) and translation, both float32
    torch::Tensor _qvec = torch::tensor({1.f, 0.f, 0.f, 0.f}, torch::kFloat32);
    torch::Tensor _tvec = torch::zeros({3}, torch::kFloat32);

private:
    uint32_t _image_ID = 0;
};
