#pragma once

#include "camera.cuh"
#include "point_cloud.cuh"
#include <camera_info.cuh>
#include <eigen3/Eigen/Dense>
#include <filesystem>

// Also here as in camera info. I guess this can be cleaned up and removed later on
// TODO: Check and remove this class if possible
struct SceneInfo {
    std::vector<CameraInfo> _cameras;
    PointCloud _point_cloud;
    double _nerf_norm_radius;
    Eigen::Vector3d _nerf_norm_translation;
    std::filesystem::path _ply_path;
};