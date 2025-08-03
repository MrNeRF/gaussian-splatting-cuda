#pragma once

#include "core/torch_shapes.hpp"
#include <filesystem>
#include <future>
#include <string>
#include <torch/torch.h>

class Camera {
public:
    Camera() = default;

    Camera(const torch::Tensor& R,
           const torch::Tensor& T,
           float FoVx, float FoVy,
           const std::string& image_name,
           const std::filesystem::path& image_path,
           int width, int height,
           int uid);

    // Delete copy, allow move
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;
    Camera(Camera&&) = default;
    Camera& operator=(Camera&&) = default;

    // Initialize GPU tensors on demand
    void initialize_cuda_tensors();

    // Load image from disk and return it
    torch::Tensor load_and_get_image(int resolution = -1);

    // Get number of bytes in the image file
    size_t get_num_bytes_from_file() const;

    // Accessors - now return const references to avoid copies
    const torch::Tensor& world_view_transform() const {
        return _world_view_transform;
    }
    const torch::Tensor& cam_position() const {
        return _cam_position;
    }

    void enable_image_caching() {
        _cache_enabled = true;
    }

    torch::Tensor K() const;

    std::tuple<float, float, float, float> get_intrinsics() const {
            const float tanfovx = std::tan(_FoVx * 0.5f);
        const float tanfovy = std::tan(_FoVy * 0.5f);
        const float fx = _image_width / (2.f * tanfovx);
        const float fy = _image_height / (2.f * tanfovy);
        const float cx = _image_width / 2.0f;
        const float cy = _image_height / 2.0f;
        return std::make_tuple(fx, fy, cx, cy);
    }

    int image_height() const noexcept { return _image_height; }
    int image_width() const noexcept { return _image_width; }
    float FoVx() const noexcept { return _FoVx; }
    float FoVy() const noexcept { return _FoVy; }
    const std::string& image_name() const noexcept { return _image_name; }
    int uid() const noexcept { return _uid; }

private:
    // IDs
    int _uid = -1;
    float _FoVx = 0.f;
    float _FoVy = 0.f;

    // Image info
    std::string _image_name;
    std::filesystem::path _image_path;
    int _image_width = 0;
    int _image_height = 0;

    // GPU tensors (computed on demand)
    torch::Tensor _world_view_transform;
    torch::Tensor _cam_position;

    // Optional image caching in VRAM
    bool _cache_enabled = false;
    torch::Tensor _image_cache = torch::empty({0});
};