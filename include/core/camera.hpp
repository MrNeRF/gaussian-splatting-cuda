#pragma once

#include "core/torch_shapes.hpp"
#include <string>
#include <torch/torch.h>

class Camera : public torch::nn::Module {
public:
    Camera(const torch::Tensor& R, // 3×3  (CPU, float32/64)
           const torch::Tensor& T, // 3    (CPU)
           float FoVx, float FoVy,
           const torch::Tensor& image, // H×W×C 0-1 float32 or uint8
           std::string image_name,
           int uid);

    // Allocate GPU copies & matrices
    void initialize_cuda_tensors();

    // --- modern accessors --------------------------------------------------
    torch::Tensor& R() { return _R; }
    torch::Tensor& T() { return _T; }
    float FoVx() const noexcept { return _FoVx; }
    float FoVy() const noexcept { return _FoVy; }
    const std::string& image_name() const noexcept { return _image_name; }
    const torch::Tensor& original_image() const { return _original_image; }
    int image_width() const noexcept { return _image_width; }
    int image_height() const noexcept { return _image_height; }

    torch::Tensor& world_view_transform();
    torch::Tensor& full_proj_transform();
    torch::Tensor& camera_center();

    int Get_image_height() const noexcept { return image_height(); }
    int Get_image_width() const noexcept { return image_width(); }
    float Get_FoVx() const noexcept { return FoVx(); }
    float Get_FoVy() const noexcept { return FoVy(); }
    torch::Tensor& Get_world_view_transform() { return world_view_transform(); }
    torch::Tensor& Get_full_proj_transform() { return full_proj_transform(); }
    torch::Tensor& Get_camera_center() { return camera_center(); }
    const torch::Tensor& Get_original_image() const { return original_image(); }

private:
    // ids
    int _uid = -1;

    // extrinsics / intrinsics
    torch::Tensor _R = torch::eye(3);
    torch::Tensor _T = torch::zeros({3});
    float _FoVx = 0.f;
    float _FoVy = 0.f;

    // image
    std::string _image_name;
    torch::Tensor _original_image; // CPU tensor
    int _image_width = 0;
    int _image_height = 0;

    // clip planes
    float _zfar = 100.f;
    float _znear = 0.01f;

    // NeRF++ translate/scale
    torch::Tensor _trans = torch::zeros({3});

    // GPU copies (filled by initialize_cuda_tensors)
    torch::Tensor _world_view_transform;
    torch::Tensor _projection_matrix;
    torch::Tensor _full_proj_transform;
    torch::Tensor _camera_center;

    bool _cuda_initialized = false;
};

// Forward decls
struct CameraInfo;
namespace gs::param {
    struct DatasetConfig;
}
