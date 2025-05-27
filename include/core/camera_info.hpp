// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once
#include "core/camera_utils.hpp"
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

    // These fields will be populated when the image is actually loaded in the dataloader
    int _img_w = 0;                     // Actual loaded image width (after potential resizing)
    int _img_h = 0;                     // Actual loaded image height (after potential resizing)
    int _channels = 0;                  // Number of channels in loaded image
    unsigned char* _img_data = nullptr; // Image data pointer (will be null until loaded)

    std::vector<double> _params;

    // Helper method to check if image data has been loaded
    bool is_image_loaded() const {
        return _img_data != nullptr;
    }

    // Helper method to load image data on demand
    void load_image_data(int resolution = -1) {
        if (is_image_loaded()) {
            return; // Already loaded
        }

        auto [data, w, h, c] = read_image(_image_path, resolution);
        _img_data = data;
        _img_w = w;
        _img_h = h;
        _channels = c;
    }

    // Helper method to free image data when no longer needed
    void free_image_data() {
        if (_img_data != nullptr) {
            free_image(_img_data);
            _img_data = nullptr;
            _img_w = 0;
            _img_h = 0;
            _channels = 0;
        }
    }

    // Destructor to ensure memory is cleaned up
    ~CameraInfo() {
        free_image_data();
    }

    // Copy constructor - don't copy image data, just the path
    CameraInfo(const CameraInfo& other)
        : _camera_ID(other._camera_ID),
          _R(other._R),
          _T(other._T),
          _fov_x(other._fov_x),
          _fov_y(other._fov_y),
          _image_name(other._image_name),
          _image_path(other._image_path),
          _camera_model(other._camera_model),
          _width(other._width),
          _height(other._height),
          _img_w(0),
          _img_h(0),
          _channels(0),
          _img_data(nullptr),
          _params(other._params) {
        // Don't copy image data - it will be loaded on demand
    }

    // Copy assignment operator
    CameraInfo& operator=(const CameraInfo& other) {
        if (this != &other) {
            // Free existing image data
            free_image_data();

            // Copy basic properties
            _camera_ID = other._camera_ID;
            _R = other._R;
            _T = other._T;
            _fov_x = other._fov_x;
            _fov_y = other._fov_y;
            _image_name = other._image_name;
            _image_path = other._image_path;
            _camera_model = other._camera_model;
            _width = other._width;
            _height = other._height;
            _params = other._params;

            // Don't copy image data - reset to unloaded state
            _img_w = 0;
            _img_h = 0;
            _channels = 0;
            _img_data = nullptr;
        }
        return *this;
    }

    // Move constructor
    CameraInfo(CameraInfo&& other) noexcept
        : _camera_ID(other._camera_ID),
          _R(std::move(other._R)),
          _T(std::move(other._T)),
          _fov_x(other._fov_x),
          _fov_y(other._fov_y),
          _image_name(std::move(other._image_name)),
          _image_path(std::move(other._image_path)),
          _camera_model(other._camera_model),
          _width(other._width),
          _height(other._height),
          _img_w(other._img_w),
          _img_h(other._img_h),
          _channels(other._channels),
          _img_data(other._img_data),
          _params(std::move(other._params)) {
        // Reset the source object
        other._img_data = nullptr;
        other._img_w = 0;
        other._img_h = 0;
        other._channels = 0;
    }

    // Move assignment operator
    CameraInfo& operator=(CameraInfo&& other) noexcept {
        if (this != &other) {
            // Free existing image data
            free_image_data();

            // Move data from other
            _camera_ID = other._camera_ID;
            _R = std::move(other._R);
            _T = std::move(other._T);
            _fov_x = other._fov_x;
            _fov_y = other._fov_y;
            _image_name = std::move(other._image_name);
            _image_path = std::move(other._image_path);
            _camera_model = other._camera_model;
            _width = other._width;
            _height = other._height;
            _img_w = other._img_w;
            _img_h = other._img_h;
            _channels = other._channels;
            _img_data = other._img_data;
            _params = std::move(other._params);

            // Reset the source object
            other._img_data = nullptr;
            other._img_w = 0;
            other._img_h = 0;
            other._channels = 0;
        }
        return *this;
    }

    // Default constructor
    CameraInfo() = default;
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