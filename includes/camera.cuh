#pragma once

#include "utils.cuh"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <string>
#include <unordered_map>
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

static const std::unordered_map<int, std::pair<CAMERA_MODEL, uint32_t>> camera_model_ids = {
    {0, {CAMERA_MODEL::SIMPLE_PINHOLE, 3}},
    {1, {CAMERA_MODEL::PINHOLE, 4}},
    {2, {CAMERA_MODEL::SIMPLE_RADIAL, 4}},
    {3, {CAMERA_MODEL::RADIAL, 5}},
    {4, {CAMERA_MODEL::OPENCV, 8}},
    {5, {CAMERA_MODEL::OPENCV_FISHEYE, 8}},
    {6, {CAMERA_MODEL::FULL_OPENCV, 12}},
    {7, {CAMERA_MODEL::FOV, 5}},
    {8, {CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE, 4}},
    {9, {CAMERA_MODEL::RADIAL_FISHEYE, 5}},
    {10, {CAMERA_MODEL::THIN_PRISM_FISHEYE, 12}},
    {11, {CAMERA_MODEL::UNDEFINED, -1}}};

class CameraInfo {
public:
    CameraInfo()
        : _image_data(nullptr),
          _image_width(0),
          _image_height(0),
          _image_channels(0),
          _camera_model(CAMERA_MODEL::UNDEFINED) {}

    // Copy constructor
    CameraInfo(const CameraInfo& other)
        : _camera_ID(other._camera_ID),
          _R(other._R),
          _T(other._T),
          _fov_x(other._fov_x),
          _fov_y(other._fov_y),
          _image_name(other._image_name),
          _image_path(other._image_path),
          _image_width(other._image_width),
          _image_height(other._image_height),
          _image_channels(other._image_channels),
          _camera_model(other._camera_model) {
        _image_data = new unsigned char[_image_width * _image_height * _image_channels];
        std::copy(other._image_data, other._image_data + _image_width * _image_height * _image_channels, _image_data);
    }

    // Copy assignment operator
    CameraInfo& operator=(const CameraInfo& other) {
        if (this != &other) {
            delete[] _image_data;

            _camera_ID = other._camera_ID;
            _R = other._R;
            _T = other._T;
            _fov_x = other._fov_x;
            _fov_y = other._fov_y;
            _image_name = other._image_name;
            _image_path = other._image_path;
            _image_width = other._image_width;
            _image_height = other._image_height;
            _image_channels = other._image_channels;
            _camera_model = other._camera_model;

            _image_data = new unsigned char[_image_width * _image_height * _image_channels];
            std::copy(other._image_data, other._image_data + _image_width * _image_height * _image_channels, _image_data);
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
          _image_width(other._image_width),
          _image_height(other._image_height),
          _image_channels(other._image_channels),
          _camera_model(other._camera_model),
          _image_data(other._image_data) {
        other._image_data = nullptr;
    }

    // Move assignment operator
    CameraInfo& operator=(CameraInfo&& other) noexcept {
        if (this != &other) {
            delete[] _image_data;

            _camera_ID = other._camera_ID;
            _R = std::move(other._R);
            _T = std::move(other._T);
            _fov_x = other._fov_x;
            _fov_y = other._fov_y;
            _image_name = std::move(other._image_name);
            _image_path = std::move(other._image_path);
            _image_width = other._image_width;
            _image_height = other._image_height;
            _image_channels = other._image_channels;
            _camera_model = other._camera_model;
            _image_data = other._image_data;

            other._image_data = nullptr;
        }
        return *this;
    }

    ~CameraInfo() {
        free_image(_image_data);
    }

    void SetImage(unsigned char* image_data, uint64_t image_width, uint64_t image_height, uint64_t image_channels) {
        _image_data = image_data;
        _image_width = image_width;
        _image_height = image_height;
        _image_channels = image_channels;
    }

    uint64_t GetImageWidth() const { return _image_width; }
    uint64_t GetImageHeight() const { return _image_height; }

public:
    uint32_t _camera_ID;
    Eigen::Matrix3d _R;
    Eigen::Vector3d _T;
    double _fov_x;
    double _fov_y;
    std::string _image_name;
    std::string _image_path;
    CAMERA_MODEL _camera_model;

private:
    uint64_t _image_width;
    uint64_t _image_height;
    uint64_t _image_channels;
    unsigned char* _image_data;
};

class Camera {
public:
    explicit Camera(CAMERA_MODEL model) : _camera_model(model) {
        _model_ID = -1;
        for (auto& it : camera_model_ids) {
            if (it.second.first == _camera_model) {
                _model_ID = it.first;
                break;
            }
        }
        if (_model_ID == -1) {
            std::cerr << "Camera model not supported!" << std::endl;
            exit(EXIT_FAILURE);
        }
        _params.resize(camera_model_ids.at(_model_ID).second);
    }

    explicit Camera(int model_ID) : _model_ID(model_ID) {
        _camera_model = camera_model_ids.at(_model_ID).first;
        _params.resize(camera_model_ids.at(_model_ID).second);
    }

    int GetModelID() const { return _model_ID; }

public:
    uint32_t _camera_ID;
    CAMERA_MODEL _camera_model;
    uint64_t _width;
    uint64_t _height;
    std::vector<double> _params;
    std::string _image_name;
    Eigen::Matrix3d _R;
    Eigen::Vector3d _T;
    double _fov_x;
    double _fov_y;
    double _z_near = 100.0;
    double z_far = 0.0;

private:
    int _model_ID;
};