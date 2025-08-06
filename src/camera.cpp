#include "core/camera.hpp"
#include "core/image_io.hpp"
#include <torch/torch.h>

using torch::indexing::None;
using torch::indexing::Slice;

namespace gs {
    static torch::Tensor world_to_view(const torch::Tensor& R, const torch::Tensor& t) {
        torch::Tensor w2c = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(R.device()));
        w2c.index_put_({Slice(0, 3), Slice(0, 3)}, R);

        w2c.index_put_({Slice(0, 3), 3}, t);

        return w2c.to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).unsqueeze(0).contiguous();
    }

    Camera::Camera(const torch::Tensor& R,
                   const torch::Tensor& T,
                   float focal_x, float focal_y,
                   float center_x, float center_y,
                   const torch::Tensor radial_distortion,
                   const torch::Tensor tangential_distortion,
                   gsplat::CameraModelType camera_model_type,
                   const std::string& image_name,
                   const std::filesystem::path& image_path,
                   int camera_width, int camera_height,
                   int uid)
        : _uid(uid),
          _focal_x(focal_x),
          _focal_y(focal_y),
          _center_x(center_x),
          _center_y(center_y),
          _radial_distortion(radial_distortion),
          _tangential_distortion(tangential_distortion),
          _camera_model_type(camera_model_type),
          _image_name(image_name),
          _image_path(image_path),
          _camera_width(camera_width),
          _camera_height(camera_height),
          _image_width(camera_width),
          _image_height(camera_height),
          _world_view_transform{world_to_view(R, T)} {

        auto c2w = torch::inverse(_world_view_transform.squeeze());
        _cam_position = c2w.index({Slice(None, 3), 3}).contiguous().squeeze();
        _FoVx = focal2fov(_focal_x, _camera_width);
        _FoVy = focal2fov(_focal_y, _camera_height);
    }

    torch::Tensor Camera::K() const {
        const auto K = torch::zeros({1, 3, 3}, _world_view_transform.options());
        float x_scale_factor = float(_image_width) / float(_camera_width);
        float y_scale_factor = float(_image_height) / float(_camera_height);
        K[0][0][0] = _focal_x * x_scale_factor;
        K[0][1][1] = _focal_y * y_scale_factor;
        K[0][0][2] = _center_x * x_scale_factor;
        K[0][1][2] = _center_y * y_scale_factor;
        K[0][2][2] = 1.0f;
        return K;
    }

    torch::Tensor Camera::load_and_get_image(int resolution) {
        if (_image_cache.size(0) > 0)
            return _image_cache;
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
                                  data, {h, w, c}, {w * c, c, 1}, pinned_options)
                                  .permute({2, 0, 1})
                                  .to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU), /*non_blocking=*/true) /
                              255.0f;

        free_image(data);
        if (_cache_enabled)
            _image_cache = image;
        return image;
    }

    size_t Camera::get_num_bytes_from_file() const {
        auto [w, h, c] = get_image_info(_image_path);
        size_t num_bytes = w * h * c * sizeof(float);
        return num_bytes;
    }
} // namespace gs