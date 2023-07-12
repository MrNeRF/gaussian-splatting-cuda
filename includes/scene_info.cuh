#pragma once

#include "camera.cuh"
#include "point_cloud.cuh"
#include <eigen3/Eigen/Dense>
#include <filesystem>

struct SceneInfo {
    std::vector<CameraInfo> cameras;
    PointCloud point_cloud;
    double nerf_norm_radius;
    Eigen::Vector3d nerf_norm_translation;
    std::filesystem::path ply_path;
    // Default constructor
    SceneInfo() = default;
    SceneInfo(std::vector<CameraInfo> cameras, PointCloud point_cloud, double nerf_norm_radius, Eigen::Vector3d nerf_norm_translation, std::filesystem::path ply_path)
        : cameras(cameras),
          point_cloud(point_cloud),
          nerf_norm_radius(nerf_norm_radius),
          nerf_norm_translation(nerf_norm_translation),
          ply_path(ply_path) {}

    // Destructor
    ~SceneInfo() = default;

    // Copy constructor
    SceneInfo(const SceneInfo& other)
        : cameras(other.cameras),
          point_cloud(other.point_cloud),
          nerf_norm_radius(other.nerf_norm_radius),
          nerf_norm_translation(other.nerf_norm_translation),
          ply_path(other.ply_path) {}

    // Copy assignment operator
    SceneInfo& operator=(const SceneInfo& other) {
        if (this != &other) {
            cameras = other.cameras;
            point_cloud = other.point_cloud;
            nerf_norm_radius = other.nerf_norm_radius;
            nerf_norm_translation = other.nerf_norm_translation;
            ply_path = other.ply_path;
        }
        return *this;
    }

    // Move constructor
    SceneInfo(SceneInfo&& other) noexcept
        : cameras(std::move(other.cameras)),
          point_cloud(std::move(other.point_cloud)),
          nerf_norm_radius(other.nerf_norm_radius),
          nerf_norm_translation(other.nerf_norm_translation),
          ply_path(std::move(other.ply_path)) {
        // Reset the moved-from object's members
        other.nerf_norm_radius = 0.0;
        other.nerf_norm_translation = Eigen::Vector3d::Zero();
    }

    // Move assignment operator
    SceneInfo& operator=(SceneInfo&& other) noexcept {
        if (this != &other) {
            cameras = std::move(other.cameras);
            point_cloud = std::move(other.point_cloud);
            nerf_norm_radius = other.nerf_norm_radius;
            nerf_norm_translation = other.nerf_norm_translation;
            ply_path = std::move(other.ply_path);

            // Reset the moved-from object's members
            other.nerf_norm_radius = 0.0;
            other.nerf_norm_translation = Eigen::Vector3d::Zero();
        }
        return *this;
    }
};