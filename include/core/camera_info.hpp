#pragma once
//
//  Lightweight container that holds per-image camera data at load time
//
#include "core/camera_utils.hpp"

#include <filesystem>
#include <string>
#include <torch/torch.h>
#include <utility>

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

struct CameraInfo {
    // ---- static data loaded from COLMAP -----------------------------------
    uint32_t _camera_ID = 0;
    torch::Tensor _R = torch::eye(3, torch::kFloat32);     // (3×3)
    torch::Tensor _T = torch::zeros({3}, torch::kFloat32); // (3)
    float _fov_x = 0.f;
    float _fov_y = 0.f;
    std::string _image_name;
    std::filesystem::path _image_path;
    CAMERA_MODEL _camera_model = CAMERA_MODEL::UNDEFINED;
    int _width = 0;
    int _height = 0;
    torch::Tensor _params; // 1-D float32

    // ---- image payload (filled lazily) ------------------------------------
    int _img_w = 0;
    int _img_h = 0;
    int _channels = 0;
    unsigned char* _img_data = nullptr;

    // -----------------------------------------------------------------------
    //  Helpers
    // -----------------------------------------------------------------------
    [[nodiscard]] bool is_image_loaded() const noexcept {
        return _img_data != nullptr;
    }

    void load_image_data(int resolution = -1) {
        if (is_image_loaded())
            return;
        auto [data, w, h, c] = read_image(_image_path, resolution);
        _img_data = data;
        _img_w = w;
        _img_h = h;
        _channels = c;
    }

    void free_image_data() {
        if (_img_data) {
            free_image(_img_data);
            _img_data = nullptr;
        }
        _img_w = _img_h = _channels = 0;
    }

    // -----------------------------------------------------------------------
    //  Rule-of-five (do NOT duplicate image buffers)
    // -----------------------------------------------------------------------
    ~CameraInfo() { free_image_data(); }

    CameraInfo() = default;

    CameraInfo(const CameraInfo& o)
        : _camera_ID(o._camera_ID),
          _R(o._R.clone()),
          _T(o._T.clone()),
          _fov_x(o._fov_x),
          _fov_y(o._fov_y),
          _image_name(o._image_name),
          _image_path(o._image_path),
          _camera_model(o._camera_model),
          _width(o._width),
          _height(o._height),
          _params(o._params.clone()) // tensors share storage → clone
    {                                /* image data left unloaded */
    }

    CameraInfo& operator=(const CameraInfo& o) {
        if (this != &o) {
            free_image_data();
            _camera_ID = o._camera_ID;
            _R = o._R.clone();
            _T = o._T.clone();
            _fov_x = o._fov_x;
            _fov_y = o._fov_y;
            _image_name = o._image_name;
            _image_path = o._image_path;
            _camera_model = o._camera_model;
            _width = o._width;
            _height = o._height;
            _params = o._params.clone();
        }
        return *this;
    }

    CameraInfo(CameraInfo&& o) noexcept
        : _camera_ID(o._camera_ID),
          _R(std::move(o._R)),
          _T(std::move(o._T)),
          _fov_x(o._fov_x),
          _fov_y(o._fov_y),
          _image_name(std::move(o._image_name)),
          _image_path(std::move(o._image_path)),
          _camera_model(o._camera_model),
          _width(o._width),
          _height(o._height),
          _params(std::move(o._params)),
          _img_w(o._img_w),
          _img_h(o._img_h),
          _channels(o._channels),
          _img_data(o._img_data) {
        o._img_data = nullptr;
        o._img_w = o._img_h = o._channels = 0;
    }

    CameraInfo& operator=(CameraInfo&& o) noexcept {
        if (this != &o) {
            free_image_data();
            _camera_ID = o._camera_ID;
            _R = std::move(o._R);
            _T = std::move(o._T);
            _fov_x = o._fov_x;
            _fov_y = o._fov_y;
            _image_name = std::move(o._image_name);
            _image_path = std::move(o._image_path);
            _camera_model = o._camera_model;
            _width = o._width;
            _height = o._height;
            _params = std::move(o._params);
            _img_w = o._img_w;
            _img_h = o._img_h;
            _channels = o._channels;
            _img_data = o._img_data;

            o._img_data = nullptr;
            o._img_w = o._img_h = o._channels = 0;
        }
        return *this;
    }
};
