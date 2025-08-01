#pragma once

#include "Common.h"
#include <filesystem>
#include <future>
#include <string>
#include <torch/torch.h>

namespace gs {

    class Camera {
    public:
        Camera() = default;

        Camera(const torch::Tensor& R,
               const torch::Tensor& T,
               float focal_x, float focal_y,
               float center_x, float center_y,
               torch::Tensor radial_distortion,
               torch::Tensor tangential_distortion,
               gsplat::CameraModelType camera_model_type,
               const std::string& image_name,
               const std::filesystem::path& image_path,
               int camera_width, int camera_height,
               int uid);
        Camera(const Camera&, const torch::Tensor& transform);

        // Delete copy, allow move
        Camera(const Camera&) = delete;
        Camera& operator=(const Camera&) = delete;
        Camera(Camera&&) = default;
        Camera& operator=(Camera&&) = default;

        // Initialize GPU tensors on demand
        void initialize_cuda_tensors();

        // Load image from disk and return it
        torch::Tensor load_and_get_image(int resolution = -1);

        // Accessors - now return const references to avoid copies
        const torch::Tensor& world_view_transform() const {
            return _world_view_transform;
        }

        torch::Tensor K() const;

        int image_height() const noexcept { return _image_height; }
        int image_width() const noexcept { return _image_width; }
        float focal_x() const noexcept { return _focal_x; }
        float focal_y() const noexcept { return _focal_y; }
        torch::Tensor radial_distortion() const noexcept { return _radial_distortion; }
        torch::Tensor tangential_distortion() const noexcept { return _tangential_distortion; }
        gsplat::CameraModelType camera_model_type() const noexcept { return _camera_model_type; }
        const std::string& image_name() const noexcept { return _image_name; }
        int uid() const noexcept { return _uid; }

    private:
        // IDs
        int _uid = -1;
        float _focal_x = 0.f;
        float _focal_y = 0.f;
        float _center_x = 0.f;
        float _center_y = 0.f;

        torch::Tensor _radial_distortion = torch::empty({0}, torch::kFloat32);
        torch::Tensor _tangential_distortion = torch::empty({0}, torch::kFloat32);
        gsplat::CameraModelType _camera_model_type = gsplat::CameraModelType::PINHOLE;

        // Image info
        std::string _image_name;
        std::filesystem::path _image_path;
        int _camera_width = 0;
        int _camera_height = 0;
        int _image_width = 0;
        int _image_height = 0;

        // GPU tensors (computed on demand)
        torch::Tensor _world_view_transform;
    };

    inline float focal2fov(float focal, int pixels) {
        return 2.0f * std::atan(pixels / (2.0f * focal));
    }

    inline float fov2focal(float fov, int pixels) {
        float tan_fov = std::tan(fov * 0.5f);
        return pixels / (2.0f * tan_fov);
    }

} // namespace gs