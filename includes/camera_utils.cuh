#pragma once

#include "camera.cuh"
#pragma diag_suppress code_of_warning
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Geometry/Quaternion.h>
#pragma diag_default code_of_warning
#include <filesystem>
#include <torch/torch.h>

torch::Tensor getWorld2View2(const Eigen::Matrix3d& R,
                             const Eigen::Vector3d& t,
                             const Eigen::Vector3d& translate = Eigen::Vector3d::Zero(),
                             float scale = 1.0);

// TODO: hacky. Find better way
Eigen::Matrix4d getWorld2View2Eigen(const Eigen::Matrix3d& R,
                                    const Eigen::Vector3d& t,
                                    const Eigen::Vector3d& translate = Eigen::Vector3d::Zero(),
                                    float scale = 1.0);

torch::Tensor getProjectionMatrix(double znear, double zfar, double fovX, double fovY);

double fov2focal(double fov, double pixels);

double focal2fov(double focal, double pixels);

Eigen::Matrix3d qvec2rotmat(const Eigen::Quaterniond& qvec);

Eigen::Quaterniond rotmat2qvec(const Eigen::Matrix3d& R);

std::tuple<unsigned char*, int, int, int> read_image(std::filesystem::path image_path);

void free_image(unsigned char* image);
