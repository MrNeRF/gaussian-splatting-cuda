#pragma once
#include "core/camera.hpp"
#include "core/point_cloud.hpp"
#include <filesystem>
#include <vector>

enum class CAMERA_MODEL {
    SIMPLE_PINHOLE = 0,
    PINHOLE = 1,
    SIMPLE_RADIAL = 2,
    RADIAL = 3,
    OPENCV = 4,
    OPENCV_FISHEYE = 5,
    FULL_OPENCV = 6,
    FOV = 7,
    SIMPLE_RADIAL_FISHEYE = 8,
    RADIAL_FISHEYE = 9,
    THIN_PRISM_FISHEYE = 10,
    UNDEFINED = 11
};

struct CameraData {
    // Static data loaded from COLMAP
    uint32_t _camera_ID = 0;
    torch::Tensor _R = torch::eye(3, torch::kFloat32);
    torch::Tensor _T = torch::zeros({3}, torch::kFloat32);
    float _fov_x = 0.f;
    float _fov_y = 0.f;
    std::string _image_name;
    std::filesystem::path _image_path;
    CAMERA_MODEL _camera_model = CAMERA_MODEL::UNDEFINED;
    int _width = 0;
    int _height = 0;
    torch::Tensor _params;

    int _img_w = 0;
    int _img_h = 0;
    int _channels = 0;
    unsigned char* _img_data = nullptr;
};

// Read COLMAP cameras, images, and compute nerf norm
std::tuple<std::vector<CameraData>, torch::Tensor> read_colmap_cameras_and_images(
    const std::filesystem::path& base,
    const std::string& images_folder = "images");

// Read COLMAP point cloud
PointCloud read_colmap_point_cloud(const std::filesystem::path& filepath);