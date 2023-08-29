#define STB_IMAGE_IMPLEMENTATION
#include "camera_utils.cuh"
#include "stb_image.h"
#include <cmath>
#include <filesystem>
#include <iostream>

torch::Tensor getWorld2View2(const Eigen::Matrix3f& R, const Eigen::Vector3f& t,
                             const Eigen::Vector3f& translate /*= Eigen::Vector3d::Zero()*/, float scale /*= 1.0*/) {
    Eigen::Matrix4f Rt = Eigen::Matrix4f::Zero();
    Rt.block<3, 3>(0, 0) = R.transpose();
    Rt.block<3, 1>(0, 3) = t;
    Rt(3, 3) = 1.0;

    Eigen::Matrix4f C2W = Rt.inverse();
    Eigen::Vector3f cam_center = C2W.block<3, 1>(0, 3);
    cam_center = (cam_center + translate) * scale;
    C2W.block<3, 1>(0, 3) = cam_center;
    Rt = C2W.inverse();
    // Here we create a torch::Tensor from the Eigen::Matrix
    // Note that the tensor will be on the CPU, you may want to move it to the desired device later
    auto RtTensor = torch::from_blob(Rt.data(), {4, 4}, torch::kFloat);
    // clone the tensor to allocate new memory, as from_blob shares the same memory
    // this step is important if Rt will go out of scope and the tensor will be used later
    return RtTensor.clone();
}

Eigen::Matrix4f getWorld2View2Eigen(const Eigen::Matrix3f& R, const Eigen::Vector3f& t,
                                    const Eigen::Vector3f& translate /*= Eigen::Vector3d::Zero()*/, float scale /*= 1.0*/) {
    Eigen::Matrix4f Rt = Eigen::Matrix4f::Zero();
    Rt.block<3, 3>(0, 0) = R.transpose();
    Rt.block<3, 1>(0, 3) = t;
    Rt(3, 3) = 1.0;

    Eigen::Matrix4f C2W = Rt.inverse();
    Eigen::Vector3f cam_center = C2W.block<3, 1>(0, 3);
    cam_center = (cam_center + translate) * scale;
    C2W.block<3, 1>(0, 3) = cam_center;
    Rt = C2W.inverse();
    return Rt;
}

torch::Tensor getProjectionMatrix(float znear, float zfar, float fovX, float fovY) {
    float tanHalfFovY = std::tan((fovY / 2.f));
    float tanHalfFovX = std::tan((fovX / 2.f));

    float top = tanHalfFovY * znear;
    float bottom = -top;
    float right = tanHalfFovX * znear;
    float left = -right;

    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();

    float z_sign = 1.f;

    P(0, 0) = 2.f * znear / (right - left);
    P(1, 1) = 2.f * znear / (top - bottom);
    P(0, 2) = (right + left) / (right - left);
    P(1, 2) = (top + bottom) / (top - bottom);
    P(3, 2) = z_sign;
    P(2, 2) = z_sign * zfar / (zfar - znear);
    P(2, 3) = -(zfar * znear) / (zfar - znear);

    // create torch::Tensor from Eigen::Matrix
    auto PTensor = torch::from_blob(P.data(), {4, 4}, torch::kFloat);

    // clone the tensor to allocate new memory
    return PTensor.clone();
}

float fov2focal(float fov, int pixels) {
    return static_cast<float>(pixels) / (2.f * std::tan(fov / 2.f));
}

float focal2fov(float focal, int pixels) {
    return 2 * std::atan(static_cast<float>(pixels) / (2.f * focal));
}

Eigen::Matrix3f qvec2rotmat(const Eigen::Quaternionf& q) {
    Eigen::Vector4f qvec = q.coeffs(); // [x, y, z, w]

    Eigen::Matrix3f rotmat;
    rotmat << 1.f - 2.f * qvec[2] * qvec[2] - 2.f * qvec[3] * qvec[3],
        2.f * qvec[1] * qvec[2] - 2.f * qvec[0] * qvec[3],
        2.f * qvec[3] * qvec[1] + 2.f * qvec[0] * qvec[2],
        2.f * qvec[1] * qvec[2] + 2.f * qvec[0] * qvec[3],
        1.f - 2.f * qvec[1] * qvec[1] - 2.f * qvec[3] * qvec[3],
        2.f * qvec[2] * qvec[3] - 2.f * qvec[0] * qvec[1],
        2.f * qvec[3] * qvec[1] - 2.f * qvec[0] * qvec[2],
        2.f * qvec[2] * qvec[3] + 2.f * qvec[0] * qvec[1],
        1.f - 2.f * qvec[1] * qvec[1] - 2.f * qvec[2] * qvec[2];

    return rotmat;
}

Eigen::Quaternionf rotmat2qvec(const Eigen::Matrix3f& R) {
    Eigen::Quaternionf qvec(R);
    // the order of coefficients is different in python implementation.
    // Might be a bug here if data comes in wrong order! TODO: check
    if (qvec.w() < 0.f) {
        qvec.coeffs() *= -1.f;
    }
    return qvec;
}

std::tuple<unsigned char*, int, int, int> read_image(std::filesystem::path image_path) {
    int width, height, channels;
    unsigned char* img = stbi_load(image_path.string().c_str(), &width, &height, &channels, 0);
    if (img == nullptr) {
        throw std::runtime_error("Could not load image " + image_path.string() + ": " + stbi_failure_reason());
    }

    return {img, width, height, channels};
}

void free_image(unsigned char* image) {
    stbi_image_free(image);
    image = nullptr;
}
