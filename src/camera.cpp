#include "core/camera.hpp"

#include <utility>

static torch::Tensor create_projection(float znear, float zfar, float fovX, float fovY) {
    float tanHalfFovY = std::tan(fovY / 2.f);
    float tanHalfFovX = std::tan(fovX / 2.f);

    float top = tanHalfFovY * znear;
    float bottom = -top;
    float right = tanHalfFovX * znear;
    float left = -right;

    torch::Tensor P = torch::zeros({4, 4}, torch::kFloat32);

    float z_sign = 1.f;

    P[0][0] = 2.f * znear / (right - left);
    P[1][1] = 2.f * znear / (top - bottom);
    P[0][2] = (right + left) / (right - left);
    P[1][2] = (top + bottom) / (top - bottom);
    P[2][2] = z_sign * zfar / (zfar - znear);
    P[2][3] = z_sign;
    P[3][2] = -(zfar * znear) / (zfar - znear);

    // Just clone, no transpose
    return P.clone();
}

static torch::Tensor world_to_view(const torch::Tensor& R, const torch::Tensor& t) {
    assert_mat(R, 3, 3, "R");
    assert_vec(t, 3, "t");

    torch::Tensor Rt = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(R.device()));

    // 1. Put R^T in the top-left block
    Rt.index_put_({torch::indexing::Slice(0, 3),
                   torch::indexing::Slice(0, 3)},
                  R.t());

    // 2. Copy translation
    Rt.index_put_({3, torch::indexing::Slice(0, 3)}, t);

    return Rt;
}

Camera::Camera(const torch::Tensor& R,
               const torch::Tensor& T,
               float FoVx, float FoVy,
               const torch::Tensor& image,
               std::string image_name,
               int uid)
    : _uid(uid),
      _FoVx(FoVx),
      _FoVy(FoVy),
      _image_name(std::move(image_name)) {
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
    auto R = _R.to(torch::kCUDA, /*non_blocking=*/true);
    auto t = _T.to(torch::kCUDA, /*non_blocking=*/true);

    assert_mat(R, 3, 3, "R");
    assert_vec(t, 3, "t");

    _world_view_transform = world_to_view(R, t);
    _projection_matrix = create_projection(_znear, _zfar, _FoVx, _FoVy).to(torch::kCUDA);

    this->_full_proj_transform = this->_world_view_transform.unsqueeze(0).bmm(this->_projection_matrix.unsqueeze(0)).squeeze(0);
    this->_camera_center = this->_world_view_transform.inverse()[3].slice(0, 0, 3);

    _cuda_initialized = true;
}

// -----------------------------------------------------------------------------
//  Accessors that guard CUDA init
// -----------------------------------------------------------------------------
torch::Tensor& Camera::world_view_transform() {
    TORCH_CHECK(_cuda_initialized, "initialize_cuda_tensors() not called");
    return _world_view_transform;
}
torch::Tensor& Camera::full_proj_transform() { return world_view_transform(), _full_proj_transform; }
torch::Tensor& Camera::camera_center() { return world_view_transform(), _camera_center; }
