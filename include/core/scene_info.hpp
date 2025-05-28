#pragma once

#include "core/camera.hpp"
#include "core/camera_info.hpp"
#include <filesystem>

struct Point {
    float x;
    float y;
    float z;
};

struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

struct PointCloud {
    std::vector<Point> _points;
    std::vector<Color> _colors;
};

struct SceneInfo {
    std::vector<CameraInfo> _cameras;
    PointCloud _point_cloud;
    float _nerf_norm_radius;
};
