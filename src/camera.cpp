#include "core/camera.hpp"
#include "core/image_io.hpp"

namespace F = torch::nn::functional;

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
      _image_width(width),
      _image_height(height),
      _original_width(width),
      _original_height(height),
      _world_view_transform{world_to_view(R, T)} {
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

torch::Tensor Camera::K_with_factor(float factor) const {
    // Compute intrinsics for scaled resolution
    const float scaled_width = _original_width * factor;
    const float scaled_height = _original_height * factor;

    const float tanfovx = std::tan(_FoVx * 0.5f);
    const float tanfovy = std::tan(_FoVy * 0.5f);
    const float focal_length_x = scaled_width / (2.f * tanfovx);
    const float focal_length_y = scaled_height / (2.f * tanfovy);

    const float cx = scaled_width / 2.0f;
    const float cy = scaled_height / 2.0f;

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

    // Load image
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
                              .clone() /
                          255.0f;

    free_image(data);
    return image.to(torch::kCUDA, /*non_blocking=*/true);
}

torch::Tensor Camera::load_and_get_image_with_factor(float factor) {
    if (factor >= 0.99f) {
        // Full resolution
        return load_and_get_image(-1);
    }

    // Load full resolution image first
    unsigned char* data;
    int w, h, c;

    auto result = load_image(_image_path, -1);
    data = std::get<0>(result);
    w = std::get<1>(result);
    h = std::get<2>(result);
    c = std::get<3>(result);

    // Convert to tensor
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

    // Move to GPU
    image = image.to(torch::kCUDA, /*non_blocking=*/true);

    // Apply downscaling if needed
    if (factor < 0.99f) {
        int new_h = static_cast<int>(h * factor);
        int new_w = static_cast<int>(w * factor);

        // Ensure minimum size
        new_h = std::max(new_h, 2);
        new_w = std::max(new_w, 2);

        // Update dimensions
        _image_width = new_w;
        _image_height = new_h;

        // Downsample using area interpolation (best for downsampling)
        image = F::interpolate(
                    image.unsqueeze(0),
                    F::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{new_h, new_w})
                        .mode(torch::kArea))
                    .squeeze(0);
    } else {
        _image_width = w;
        _image_height = h;
    }

    return image;
}