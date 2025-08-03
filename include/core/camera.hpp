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
    torch::Tensor load_image_cache(int resolution = -1);

    void set_image_height(const int& height) { _image_height = height;}
    void set_image_width(const int& width) { _image_width = width; }

    // Accessors - now return const references to avoid copies
    const torch::Tensor& world_view_transform() const {
        return _world_view_transform;
    }

    const torch::Tensor& projmat() const {
        return _projmat;
    }

    const torch::Tensor& campos() const {
        return _campos;
    }

    torch::Tensor K() const;

    int image_height() const noexcept { return _image_height; }
    int image_width() const noexcept { return _image_width; }
    float FoVx() const noexcept { return _FoVx; }
    float FoVy() const noexcept { return _FoVy; }
    float tanfovx() const noexcept { return _tanfovx; }
    float tanfovy() const noexcept { return _tanfovy; }
    const std::string& image_name() const noexcept { return _image_name; }
    int uid() const noexcept { return _uid; }

    // Variables to determine where to cache the image tensors
    bool cache_on_gpu = false;
    bool cache_on_cpu = false;

private:
    // IDs
    int _uid = -1;
    float _FoVx = 0.f;
    float _FoVy = 0.f;
    float _tanfovx = 0.f;
    float _tanfovy = 0.f;

    // Image info
    std::string _image_name;
    std::filesystem::path _image_path;
    int _image_width = 0;
    int _image_height = 0;

    // GPU tensors (computed on demand)
    torch::Tensor _world_view_transform;
    torch::Tensor _projmat;
    torch::Tensor _campos;

    // Dynamically cache image tensor
    torch::Tensor _image_cache;
};