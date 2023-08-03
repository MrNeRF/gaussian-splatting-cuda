#include "camera.cuh"
#include "camera_utils.cuh"

CameraInfo::CameraInfo()
    : _image_data(nullptr),
      _image_width(0),
      _image_height(0),
      _image_channels(0),
      _camera_model(CAMERA_MODEL::UNDEFINED) {
}

CameraInfo::CameraInfo(const CameraInfo& other)
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

CameraInfo& CameraInfo::operator=(const CameraInfo& other) {
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

CameraInfo::CameraInfo(CameraInfo&& other) noexcept
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

CameraInfo& CameraInfo::operator=(CameraInfo&& other) noexcept {
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

CameraInfo::~CameraInfo() {
    free_image(_image_data);
}
void CameraInfo::SetImage(unsigned char* image_data, uint64_t image_width, uint64_t image_height, uint64_t image_channels) {
    _image_data = image_data;
    _image_width = image_width;
    _image_height = image_height;
    _image_channels = image_channels;
}

Camera::Camera(CAMERA_MODEL model) : _camera_model(model) {
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

Camera::Camera(int model_ID) : _model_ID(model_ID) {
    _camera_model = camera_model_ids.at(_model_ID).first;
    _params.resize(camera_model_ids.at(_model_ID).second);
}
