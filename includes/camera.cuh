#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

class Camera : torch::nn::Module {
public:
    explicit Camera(int model_ID);
    int GetModelID() const { return _model_ID; }

public:
    uint32_t _camera_ID;
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