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
    // Getters
    int Get_uid() const { return _uid; }
    int Get_colmap_id() const { return _colmap_id; }
    Eigen::Matrix3d& Get_R() { return _R; }
    Eigen::Vector3d& Get_T() { return _T; }
    float Get_FoVx() const { return static_cast<float>(_FoVx); }
    float Get_FoVy() const { return static_cast<float>(_FoVy); }
    std::string Get_image_name() const { return _image_name; }
    const torch::Tensor& Get_original_image() { return _original_image; }
    int Get_image_width() const { return _image_width; }
    int Get_image_height() const { return _image_height; }
    double Get_zfar() const { return _zfar; }
    double Get_znear() const { return _znear; }
    torch::Tensor& Get_world_view_transform() { return _world_view_transform; }
    torch::Tensor& Get_projection_matrix() { return _projection_matrix; }
    torch::Tensor& Get_full_proj_transform() { return _full_proj_transform; }
    torch::Tensor& Get_camera_center() { return _camera_center; }

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
