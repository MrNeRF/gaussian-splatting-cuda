#pragma once

#include "camera.cuh"
#include "point_cloud.cuh"
#include <eigen3/Eigen/Dense>
#include <filesystem>

struct SceneInfo {
    std::vector<CameraInfo> _cameras;
    PointCloud _point_cloud;
    double _nerf_norm_radius;
    Eigen::Vector3d _nerf_norm_translation;
    std::filesystem::path _ply_path;
    // Default constructor
    SceneInfo() = default;
    SceneInfo(std::vector<CameraInfo> cameras, PointCloud point_cloud, double nerf_norm_radius, Eigen::Vector3d nerf_norm_translation, std::filesystem::path ply_path)
        : _cameras(cameras),
          _point_cloud(point_cloud),
          _nerf_norm_radius(nerf_norm_radius),
          _nerf_norm_translation(nerf_norm_translation),
          _ply_path(ply_path) {}

    // Destructor
    ~SceneInfo() = default;

    // Copy constructor
    SceneInfo(const SceneInfo& other)
        : _cameras(other._cameras),
          _point_cloud(other._point_cloud),
          _nerf_norm_radius(other._nerf_norm_radius),
          _nerf_norm_translation(other._nerf_norm_translation),
          _ply_path(other._ply_path) {}

    // Copy assignment operator
    SceneInfo& operator=(const SceneInfo& other) {
        if (this != &other) {
            _cameras = other._cameras;
            _point_cloud = other._point_cloud;
            _nerf_norm_radius = other._nerf_norm_radius;
            _nerf_norm_translation = other._nerf_norm_translation;
            _ply_path = other._ply_path;
        }
        return *this;
    }

    // Move constructor
    SceneInfo(SceneInfo&& other) noexcept
        : _cameras(std::move(other._cameras)),
          _point_cloud(std::move(other._point_cloud)),
          _nerf_norm_radius(other._nerf_norm_radius),
          _nerf_norm_translation(other._nerf_norm_translation),
          _ply_path(std::move(other._ply_path)) {
        // Reset the moved-from object's members
        other._nerf_norm_radius = 0.0;
        other._nerf_norm_translation = Eigen::Vector3d::Zero();
    }

    // Move assignment operator
    SceneInfo& operator=(SceneInfo&& other) noexcept {
        if (this != &other) {
            _cameras = std::move(other._cameras);
            _point_cloud = std::move(other._point_cloud);
            _nerf_norm_radius = other._nerf_norm_radius;
            _nerf_norm_translation = other._nerf_norm_translation;
            _ply_path = std::move(other._ply_path);

            // Reset the moved-from object's members
            other._nerf_norm_radius = 0.0;
            other._nerf_norm_translation = Eigen::Vector3d::Zero();
        }
        return *this;
    }
};