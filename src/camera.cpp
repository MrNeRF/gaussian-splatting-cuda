#include "core/camera.hpp"
#include "core/image_io.hpp"

using namespace torch::indexing;

static torch::Tensor world_to_view(const torch::Tensor& R, const torch::Tensor& t) {
    assert_mat(R, 3, 3, "R");
    assert_vec(t, 3, "t");

    torch::Tensor Rt = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(R.device()));

    Rt.index_put_({torch::indexing::Slice(0, 3),
                   torch::indexing::Slice(0, 3)},
                  R.t());

    Rt.index_put_({3, torch::indexing::Slice(0, 3)}, t);

    auto pinned_options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
    return Rt.t().unsqueeze(0).to(pinned_options);
}

static torch::Tensor compute_projmat(
    const float fx, const float fy,
    const int image_width, const int image_height, const torch::Tensor& viewmat,
    const float near_plane = 0.01f, const float far_plane = 100.0f) {

    const float cx = image_width / 2.0f;
    const float cy = image_height / 2.0f;

    float top = image_height / (2.0f * fy) * near_plane;
    float bottom = -top;
    float right = image_width / (2.0f * fx) * near_plane;
    float left = -right;

    const auto P = torch::zeros({4, 4}, viewmat.options());
    P[0][0] = 2.0f * near_plane / (right - left);
    P[1][1] = 2.0f * near_plane / (top - bottom);
    P[2][2] = far_plane / (far_plane - near_plane);
    P[3][2] = 1.0f;
    P[2][3] = -far_plane * near_plane / (far_plane - near_plane);
    P.unsqueeze(0);
    auto projmat = torch::matmul(P, viewmat);

    auto pinned_options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
    return projmat.to(pinned_options);
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
      _tanfovx(std::tan(FoVx * 0.5f)),
      _tanfovy(std::tan(FoVy * 0.5f)),
      _image_name(image_name),
      _image_path(image_path),
      _image_width(width),
      _image_height(height),
      _world_view_transform{world_to_view(R, T)} {
        const float fx = _image_width / (2.f * _tanfovx);
        const float fy = _image_height / (2.f * _tanfovy);
        _projmat = compute_projmat(fx, fy, width, height, _world_view_transform);
        _campos = torch::inverse(_world_view_transform).index({Slice(), Slice(0, 3), 3});
}

torch::Tensor Camera::K() const {
    const float tanfovx = std::tan(_FoVx * 0.5f);
    const float tanfovy = std::tan(_FoVy * 0.5f);
    const float focal_length_x = _image_width / (2.f * tanfovx);
    const float focal_length_y = _image_height / (2.f * tanfovy);

    const float cx = _image_width / 2.0f;
    const float cy = _image_height / 2.0f;

    const auto K = torch::zeros({1, 3, 3}, _world_view_transform.options());
    K[0][0][0] = focal_length_x;
    K[0][1][1] = focal_length_y;
    K[0][0][2] = cx;
    K[0][1][2] = cy;
    K[0][2][2] = 1.0f;
    return K;
}

torch::Tensor Camera::load_and_get_image(int resolution) {
    unsigned char* data;
    int w, h, c;

    // Otherwise load synchronously
    auto result = load_image(_image_path, resolution);
    data = std::get<0>(result);
    w = std::get<1>(result);
    h = std::get<2>(result);
    c = std::get<3>(result);

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
                              .contiguous()
                              .pin_memory() /
                          255.0f;

    free_image(data);
    return image;
}

torch::Tensor Camera::load_image_cache(int resolution) {

    if (cache_on_gpu) {
        if (!_image_cache.defined()) {
            _image_cache = load_and_get_image(resolution).to(torch::kCUDA, /*non_blocking=*/true);
        }
        return _image_cache;
    } else if (cache_on_cpu) {
        if (!_image_cache.defined()) {
            _image_cache = load_and_get_image(resolution);
        }
        return _image_cache.to(torch::kCUDA, /*non_blocking=*/true);
    }

    return load_and_get_image(resolution).to(torch::kCUDA, /*non_blocking=*/true);

}