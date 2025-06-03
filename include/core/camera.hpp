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

    // Accessors - now return const references to avoid copies
    const torch::Tensor& world_view_transform() const {
        return _world_view_transform;
    }

    torch::Tensor K() const;

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
};