#include "core/camera.hpp"
#include "core/camera_utils.hpp"

#include <utility>

Camera::Camera(int imported_colmap_id,
               const torch::Tensor& R,
               const torch::Tensor& T,
               float FoVx, float FoVy,
               const torch::Tensor& image,
               std::string image_name,
               int uid,
               float scale)
    : _uid(uid),
      _colmap_id(imported_colmap_id),
      _FoVx(FoVx),
      _FoVy(FoVy),
      _image_name(std::move(image_name)),
      _scale(scale) {
    assert_mat(R, 3, 3, "R");
    assert_vec(T, 3, "T");

    _R = R.to(torch::kFloat32).clone();
    _T = T.to(torch::kFloat32).clone();

    _original_image = torch::clamp(image.to(torch::kFloat32), 0.f, 1.f).contiguous();
    _image_height = static_cast<int>(_original_image.size(1));
    _image_width = static_cast<int>(_original_image.size(2));
}

// -----------------------------------------------------------------------------
//  GPU initialisation
// -----------------------------------------------------------------------------
void Camera::initialize_cuda_tensors() {
    if (_cuda_initialized)
        return;

    // move R,T once; everything below stays on cuda:0
    auto R_cuda = _R.to(torch::kCUDA, /*non_blocking=*/true);
    auto T_cuda = _T.to(torch::kCUDA, /*non_blocking=*/true);

    _world_view_transform =
        getWorld2View2(R_cuda,
                       T_cuda,
                       torch::zeros({3}, torch::TensorOptions()
                                             .dtype(torch::kFloat32)
                                             .device(torch::kCUDA)), // <-- same device
                       _scale);

    _projection_matrix =
        getProjectionMatrix(_znear, _zfar, _FoVx, _FoVy)
            .to(torch::kCUDA, /*non_blocking=*/true);

    _full_proj_transform = _projection_matrix.matmul(_world_view_transform);

    auto WV_inv = torch::linalg_inv(_world_view_transform);
    _camera_center = WV_inv.index({torch::indexing::Slice(0, 3), 3});

    _cuda_initialized = true;
}

// -----------------------------------------------------------------------------
//  Accessors that guard CUDA init
// -----------------------------------------------------------------------------
torch::Tensor& Camera::world_view_transform() {
    TORCH_CHECK(_cuda_initialized, "initialize_cuda_tensors() not called");
    return _world_view_transform;
}
torch::Tensor& Camera::projection_matrix() { return world_view_transform(), _projection_matrix; }
torch::Tensor& Camera::full_proj_transform() { return world_view_transform(), _full_proj_transform; }
torch::Tensor& Camera::camera_center() { return world_view_transform(), _camera_center; }
