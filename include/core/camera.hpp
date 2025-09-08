/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "Common.h"
#include <c10/cuda/CUDAStream.h>
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
        torch::Tensor load_and_get_image(int resize_factor = -1);

        // Load image from disk just to populate _image_width/_image_height
        void load_image_size(int resize_factor = -1);

        // Get number of bytes in the image file
        size_t get_num_bytes_from_file() const;

        // Accessors - now return const references to avoid copies
        const torch::Tensor& world_view_transform() const {
            return _world_view_transform;
        }
        const torch::Tensor& cam_position() const {
            return _cam_position;
        }

        const torch::Tensor& R() const { return _R; }
        const torch::Tensor& T() const { return _T; }

        torch::Tensor K() const;

        std::tuple<float, float, float, float> get_intrinsics() const;

        int image_height() const noexcept { return _image_height; }
        int image_width() const noexcept { return _image_width; }
        int camera_height() const noexcept { return _camera_height; }
        int camera_width() const noexcept { return _camera_width; }
        void update_image_dimensions(int width, int height) {_image_width = width; _image_height = height;}
        float focal_x() const noexcept { return _focal_x; }
        float focal_y() const noexcept { return _focal_y; }
        torch::Tensor radial_distortion() const noexcept { return _radial_distortion; }
        torch::Tensor tangential_distortion() const noexcept { return _tangential_distortion; }
        gsplat::CameraModelType camera_model_type() const noexcept { return _camera_model_type; }
        const std::string& image_name() const noexcept { return _image_name; }
        const std::filesystem::path& image_path() const noexcept { return _image_path; }
        int uid() const noexcept { return _uid; }

        float FoVx() const noexcept { return _FoVx; }
        float FoVy() const noexcept { return _FoVy; }

    private:
        // IDs
        float _FoVx = 0.f;
        float _FoVy = 0.f;
        int _uid = -1;
        float _focal_x = 0.f;
        float _focal_y = 0.f;
        float _center_x = 0.f;
        float _center_y = 0.f;

        // redundancy with _world_view_transform, but save calculation and passing from GPU 2 CPU
        torch::Tensor _R;
        torch::Tensor _T;

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
        torch::Tensor _cam_position;

        // CUDA stream for async operations
        at::cuda::CUDAStream _stream = at::cuda::getStreamFromPool(false);
    };

    inline float focal2fov(float focal, int pixels) {
        return 2.0f * std::atan(pixels / (2.0f * focal));
    }

    inline float fov2focal(float fov, int pixels) {
        float tan_fov = std::tan(fov * 0.5f);
        return pixels / (2.0f * tan_fov);
    }

} // namespace gs
