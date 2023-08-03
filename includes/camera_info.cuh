// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once
#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <vector>

enum class CAMERA_MODEL {
    SIMPLE_PINHOLE = 0,
    PINHOLE = 1,
    SIMPLE_RADIAL = 2,
    RADIAL = 3,
    OPENCV = 4,
    OPENCV_FISHEYE = 5,
    FULL_OPENCV = 6,
    FOV = 7,
    SIMPLE_RADIAL_FISHEYE = 8,
    RADIAL_FISHEYE = 9,
    THIN_PRISM_FISHEYE = 10,
    UNDEFINED = 11
};
// This class stores all information about a camera at loading time
// To me this seems to be double work, since we already have a Camera class
// I guess this can be removed later on
// TODO: Check and remove this struct if possible
struct CameraInfo {
    uint32_t _camera_ID;
    Eigen::Matrix3d _R; // rotation  matrix
    Eigen::Vector3d _T; // translation vector
    float _fov_x;
    float _fov_y;
    std::string _image_name;
    std::filesystem::path _image_path;
    CAMERA_MODEL _camera_model;
    uint64_t _image_width;
    uint64_t _image_height;
    std::vector<double> _params;
    uint64_t _image_channels;
    unsigned char* _image_data; // shallow copy is fine here. No ownership
};

// void dump_JSON(const std::filesystem::path& file_path, const nlohmann::json& json_data) {
//     std::ofstream file(file_path.string());
//     if (file.is_open()) {
//         file << json_data.dump(4); // Write the JSON data with indentation of 4 spaces
//         file.close();
//     } else {
//         throw std::runtime_error("Could not open file " + file_path.string());
//     }
// }
//
//// serialize camera to json
// nlohmann::json Dump_camera_to_JSON(Camera cam) {
//
//     Eigen::Matrix4d Rt = Eigen::Matrix4d::Zero();
//     Rt.block<3, 3>(0, 0) = cam._R.transpose();
//     Rt.block<3, 1>(0, 3) = cam._T;
//     Rt(3, 3) = 1.0;
//
//     Eigen::Matrix4d W2C = Rt.inverse();
//     Eigen::Vector3d pos = W2C.block<3, 1>(0, 3);
//     Eigen::Matrix3d rot = W2C.block<3, 3>(0, 0);
//     std::vector<std::vector<double>> serializable_array_2d;
//     for (int i = 0; i < rot.rows(); i++) {
//         serializable_array_2d.push_back(std::vector<double>(rot.row(i).data(), rot.row(i).data() + rot.row(i).size()));
//     }
//
//     nlohmann::json camera_entry = {
//         {"id", cam._camera_ID},
//         {"img_name", cam._image_name},
//         {"width", cam._width},
//         {"height", cam._height},
//         {"position", std::vector<double>(pos.data(), pos.data() + pos.size())},
//         {"rotation", serializable_array_2d},
//         {"fy", fov2focal(cam._fov_y, cam._height)},
//         {"fx", fov2focal(cam._fov_x, cam._width)}};
//
//     return camera_entry;
// }
