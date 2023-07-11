#pragma once

#include <string>
#include <unordered_map>
#include <vector>

static const std::unordered_map<int, std::pair<std::string, uint32_t>> camera_model_ids = {
    {0, {"SIMPLE_PINHOLE", 3}},
    {1, {"PINHOLE", 4}},
    {2, {"SIMPLE_RADIAL", 4}},
    {3, {"RADIAL", 5}},
    {4, {"OPENCV", 8}},
    {5, {"OPENCV_FISHEYE", 8}},
    {6, {"FULL_OPENCV", 12}},
    {7, {"FOV", 5}},
    {8, {"SIMPLE_RADIAL_FISHEYE", 4}},
    {9, {"RADIAL_FISHEYE", 5}},
    {10, {"THIN_PRISM_FISHEYE", 12}}};

class Camera {
public:
    explicit Camera(int model_ID) : _model_ID(model_ID) {
        _model_name = camera_model_ids.at(_model_ID).first;
        _params.resize(camera_model_ids.at(_model_ID).second);
    }

    int GetModelID() const { return _model_ID; }

public:
    uint32_t _camera_ID;
    std::string _model_name;
    uint64_t _width;
    uint64_t _height;
    std::vector<double> _params;

private:
    int _model_ID;
};