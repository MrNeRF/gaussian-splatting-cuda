#pragma once

#include <eigen3/Eigen/Dense>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

class Camera : torch::nn::Module {
public:
    Camera(int imported_colmap_id,
           Eigen::Matrix3d R, Eigen::Vector3d T,
           double FoVx, double FoVy,
           torch::Tensor image,
           std::string image_name,
           int image_id,
           double scale = 1.0);

private:
    int _uid;
    int _colmap_id;
    Eigen::Matrix3d _R; // rotation  matrix
    Eigen::Vector3d _T; // translation vector
    double _FoVx;
    double _FoVy;
    std::string _image_name;
    torch::Tensor _original_image;
    int _image_width;
    int _image_height;
    double _zfar;
    double _znear;
    torch::Tensor _trans;
    double _scale;
    torch::Tensor _world_view_transform;
    torch::Tensor _projection_matrix;
    torch::Tensor _full_proj_transform;
    torch::Tensor _camera_center;
};

struct CameraInfo;
struct ModelParameters;
Camera loadCam(const ModelParameters& params, int id, CameraInfo& cam_info);
