#pragma once

#include "camera.cuh"
#pragma diag_suppress code_of_warning
#include <Eigen/Dense>
#include <Eigen/src/Geometry/Quaternion.h>
#pragma diag_default code_of_warning
#include <filesystem>
#include <torch/torch.h>

torch::Tensor getWorld2View2(const Eigen::Matrix3f& R,
                             const Eigen::Vector3f& t,
                             const Eigen::Vector3f& translate = Eigen::Vector3f::Zero(),
                             float scale = 1.0);

// TODO: hacky. Find better way
Eigen::Matrix4f getWorld2View2Eigen(const Eigen::Matrix3f& R,
                                    const Eigen::Vector3f& t,
                                    const Eigen::Vector3f& translate = Eigen::Vector3f::Zero(),
                                    float scale = 1.0);

torch::Tensor getProjectionMatrix(float znear, float zfar, float fovX, float fovY);

float fov2focal(float fov, int pixels);

float focal2fov(float focal, int pixels);

Eigen::Matrix3f qvec2rotmat(const Eigen::Quaternionf& qvec);

Eigen::Quaternionf rotmat2qvec(const Eigen::Matrix3f& R);

std::tuple<unsigned char*, int, int, int> read_image(std::filesystem::path image_path);

void free_image(unsigned char* image);
