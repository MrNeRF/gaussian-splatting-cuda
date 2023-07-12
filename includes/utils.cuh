#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Geometry/Quaternion.h>
#include <filesystem>

Eigen::Matrix4f getWorld2View(const Eigen::Matrix3f& R, const Eigen::Vector3f& t);

Eigen::Matrix4f getWorld2View2(const Eigen::Matrix3f& R, const Eigen::Vector3f& t,
                               const Eigen::Vector3f& translate = Eigen::Vector3f::Zero(), float scale = 1.0);

Eigen::Matrix4f getProjectionMatrix(float znear, float zfar, float fovX, float fovY);

float fov2focal(float fov, float pixels);

float focal2fov(float focal, float pixels);

Eigen::Matrix3d qvec2rotmat(const Eigen::Quaterniond& qvec);

Eigen::Quaterniond rotmat2qvec(const Eigen::Matrix3d& R);

unsigned char* read_image(std::filesystem::path image_path, int width, int height, int channels);

void free_image(unsigned char* image);