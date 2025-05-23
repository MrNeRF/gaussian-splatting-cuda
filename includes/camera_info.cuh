// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once
#include "camera_utils.cuh"
#include <algorithm>
#pragma diag_suppress code_of_warning
#include <Eigen/Dense>
#pragma diag_default code_of_warning
#include <filesystem>
#include <fstream>
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
    Eigen::Matrix3f _R; // rotation  matrix
    Eigen::Vector3f _T; // translation vector
    float _fov_x;
    float _fov_y;
    std::string _image_name;
    std::filesystem::path _image_path;
    CAMERA_MODEL _camera_model;
    int _width;
    int _height;
    int _img_w;
    int _img_h;
    int _channels;
    std::vector<double> _params;
    unsigned char* _img_data; // shallow copy is fine here. No ownership
};

inline void dump_JSON(const std::filesystem::path& file_path, std::vector<nlohmann::json>& json_data) {
    // Ensure the directory exists
    std::filesystem::create_directories(file_path.parent_path());
    json_data.erase(std::remove_if(json_data.begin(), json_data.end(),
                                   [](const nlohmann::json& entry) { return entry.is_null(); }),
                    json_data.end());

    nlohmann::json json = json_data;

    std::ofstream file(file_path.string());
    if (file.is_open()) {
        file << json.dump(4); // Write the JSON data with indentation of 4 spaces
        file.close();
    } else {
        throw std::runtime_error("Could not open file " + file_path.string());
    }
}

// serialize camera to json
inline nlohmann::json Convert_camera_to_JSON(const CameraInfo& cam, int id, Eigen::Matrix3f& R, const Eigen::Vector3f& T) {

    Eigen::Matrix4f Rt = Eigen::Matrix4f::Zero();
    Rt.block<3, 3>(0, 0) = R.transpose();
    Rt.block<3, 1>(0, 3) = T;
    Rt(3, 3) = 1.0;

    Eigen::Matrix4f W2C = Rt.inverse();
    Eigen::Vector3f pos = W2C.block<3, 1>(0, 3);
    Eigen::Matrix3f rot = W2C.block<3, 3>(0, 0);
    std::vector<std::vector<float>> serializable_array_2d;
    for (int i = 0; i < rot.rows(); i++) {
        serializable_array_2d.push_back(std::vector<float>(rot.row(i).data(), rot.row(i).data() + rot.row(i).size()));
    }

    nlohmann::json camera_entry = {
        {"id", id},
        {"img_name", cam._image_name},
        {"width", cam._width},
        {"height", cam._height},
        {"position", std::vector<float>(pos.data(), pos.data() + pos.size())},
        {"rotation", serializable_array_2d},
        {"fy", fov2focal(cam._fov_y, cam._height)},
        {"fx", fov2focal(cam._fov_x, cam._width)}};

    return camera_entry;
}
