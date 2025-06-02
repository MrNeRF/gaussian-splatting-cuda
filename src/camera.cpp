#include "core/camera.hpp"
#include "core/image_io.hpp"
#include <cmath>

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

    return P.clone();
}

static torch::Tensor world_to_view(const torch::Tensor& R, const torch::Tensor& t) {
    assert_mat(R, 3, 3, "R");
    assert_vec(t, 3, "t");

    torch::Tensor Rt = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(R.device()));

    Rt.index_put_({torch::indexing::Slice(0, 3),
                   torch::indexing::Slice(0, 3)},
                  R.t());

    Rt.index_put_({3, torch::indexing::Slice(0, 3)}, t);

    return Rt;
}

Camera::Camera(const torch::Tensor& R,
               const torch::Tensor& T,
               float FoVx, float FoVy,
               const std::string& image_name,
               const std::filesystem::path& image_path,
               int width, int height,
               int uid)
    : _uid(uid),
      _FoVx(FoVx),
      _FoVy(FoVy),
      _image_name(image_name),
      _image_path(image_path),
      _width(width),
      _height(height) {

    assert_mat(R, 3, 3, "R");
    assert_vec(T, 3, "T");

    // Use pinned memory for CPU tensors to enable faster transfers
    auto pinned_options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);

    _R = R.to(pinned_options).clone();
    _T = T.to(pinned_options).clone();
}

void Camera::initialize_cuda_tensors() {
    // Only initialize once
    if (_cuda_initialized)
        return;

    // Use non-blocking transfers with pinned memory
    auto R = _R.to(torch::kCUDA, /*non_blocking=*/true);
    auto t = _T.to(torch::kCUDA, /*non_blocking=*/true);

    assert_mat(R, 3, 3, "R");
    assert_vec(t, 3, "t");

    _world_view_transform = world_to_view(R, t);
    _projection_matrix = create_projection(_znear, _zfar, _FoVx, _FoVy).to(torch::kCUDA);

    _full_proj_transform = _world_view_transform.unsqueeze(0).bmm(_projection_matrix.unsqueeze(0)).squeeze(0);
    _camera_center = _world_view_transform.inverse()[3].slice(0, 0, 3);

    _cuda_initialized = true;
}

void Camera::prefetch_image(int resolution) {
    if (!_image_future.valid()) {
        _image_future = load_image_async(_image_path, resolution);
    }
}

torch::Tensor Camera::load_and_get_image(int resolution) {
    unsigned char* data;
    int w, h, c;

    // If we have a prefetched future, use it
    if (_image_future.valid()) {
        auto result = _image_future.get();
        data = std::get<0>(result);
        w = std::get<1>(result);
        h = std::get<2>(result);
        c = std::get<3>(result);
    } else {
        // Otherwise load synchronously
        auto result = load_image(_image_path, resolution);
        data = std::get<0>(result);
        w = std::get<1>(result);
        h = std::get<2>(result);
        c = std::get<3>(result);
    }

    _image_width = w;
    _image_height = h;

    // Use pinned memory for faster GPU transfer
    auto pinned_options = torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true);

    torch::Tensor image = torch::from_blob(
                              data,
                              {h, w, c},
                              {w * c, c, 1},
                              pinned_options)
                              .to(torch::kFloat32)
                              .permute({2, 0, 1})
                              .clone() /
                          255.0f;

    free_image(data);
    return image.to(torch::kCUDA, /*non_blocking=*/true);
}