#define STB_IMAGE_IMPLEMENTATION
#include "utils.cuh"
#include "stb_image.h"
#include <cmath>

Eigen::Matrix4f getWorld2View(const Eigen::Matrix3f& R, const Eigen::Vector3f& t) {
    Eigen::Matrix4f Rt = Eigen::Matrix4f::Zero();
    Rt.block<3, 3>(0, 0) = R.transpose();
    Rt.block<3, 1>(0, 3) = t;
    Rt(3, 3) = 1.0;
    return Rt;
}

Eigen::Matrix4f getWorld2View2(const Eigen::Matrix3f& R, const Eigen::Vector3f& t,
                               const Eigen::Vector3f& translate /*= Eigen::Vector3f::Zero()*/, float scale /*= 1.0 */) {
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

Eigen::Matrix4f getProjectionMatrix(float znear, float zfar, float fovX, float fovY) {
    float tanHalfFovY = std::tan((fovY / 2));
    float tanHalfFovX = std::tan((fovX / 2));

    float top = tanHalfFovY * znear;
    float bottom = -top;
    float right = tanHalfFovX * znear;
    float left = -right;

    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();

    float z_sign = 1.0;

    P(0, 0) = 2.0 * znear / (right - left);
    P(1, 1) = 2.0 * znear / (top - bottom);
    P(0, 2) = (right + left) / (right - left);
    P(1, 2) = (top + bottom) / (top - bottom);
    P(3, 2) = z_sign;
    P(2, 2) = z_sign * zfar / (zfar - znear);
    P(2, 3) = -(zfar * znear) / (zfar - znear);
    return P;
}

float fov2focal(float fov, float pixels) {
    return pixels / (2 * std::tan(fov / 2));
}

float focal2fov(float focal, float pixels) {
    return 2 * std::atan(pixels / (2 * focal));
}

Eigen::Matrix3d qvec2rotmat(const Eigen::Quaterniond& qvec) {
    return qvec.normalized().toRotationMatrix();
}

Eigen::Quaterniond rotmat2qvec(const Eigen::Matrix3d& R) {
    Eigen::Quaterniond qvec(R);
    // the order of coefficients is different in python implementation.
    // Might be a bug here if data comes in wrong order! TODO: check
    if (qvec.w() < 0) {
        qvec.coeffs() *= -1;
    }
    return qvec;
}


unsigned char* read_image(std::filesystem::path image_path, int width, int height, int channels) {
    unsigned char* img = stbi_load(image_path.string().c_str(), &width, &height, &channels, 0);
    if (img == nullptr) {
        throw std::runtime_error("Could not load image: " + image_path.string());
    }

    return img;
}

void free_image(unsigned char* image) {
    stbi_image_free(image);
    image = nullptr;
}

