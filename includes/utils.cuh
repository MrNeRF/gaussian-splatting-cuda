#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Geometry/Quaternion.h>
#include <filesystem>

Eigen::Matrix4d getWorld2View(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

Eigen::Matrix4d getWorld2View2(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                               const Eigen::Vector3d& translate = Eigen::Vector3d::Zero(), float scale = 1.0);

Eigen::Matrix4d getProjectionMatrix(double znear, double zfar, double fovX, double fovY);

double fov2focal(double fov, double pixels);

double focal2fov(double focal, double pixels);

Eigen::Matrix3d qvec2rotmat(const Eigen::Quaterniond& qvec);

Eigen::Quaterniond rotmat2qvec(const Eigen::Matrix3d& R);

unsigned char* read_image(std::filesystem::path image_path, int width, int height, int channels);

void free_image(unsigned char* image);