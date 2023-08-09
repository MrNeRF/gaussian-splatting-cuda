#include "camera.cuh"
#include "camera_info.cuh"
#include "camera_utils.cuh"
#include "parameters.cuh"
#include <eigen3/Eigen/Dense>
#include <string>
#include <torch/torch.h>

Camera::Camera(int imported_colmap_id,
               Eigen::Matrix3d R, Eigen::Vector3d T,
               double FoVx, double FoVy,
               torch::Tensor image,
               std::string image_name,
               int uid,
               double scale) : _colmap_id(imported_colmap_id),
                               _R(R),
                               _T(T),
                               _FoVx(FoVx),
                               _FoVy(FoVy),
                               _image_name(image_name),
                               _uid(uid),
                               _scale(scale) {

    this->_original_image = torch::clamp(image, 0.0, 1.0).to(torch::kCUDA);
    this->_image_width = this->_original_image.size(2);
    this->_image_height = this->_original_image.size(1);

    // no masking so far
    this->_original_image *= torch::ones({1, this->_image_height, this->_image_width}).to(torch::kCUDA);

    this->_zfar = 100.0;
    this->_znear = 0.01;

    this->_world_view_transform = getWorld2View2(R, T, Eigen::Vector3d::Zero(), _scale).to(torch::kCUDA);
    this->_projection_matrix = getProjectionMatrix(this->_znear, this->_zfar, this->_FoVx, this->_FoVy).to(torch::kCUDA);
    this->_full_proj_transform = this->_world_view_transform.unsqueeze(0).bmm(this->_projection_matrix.unsqueeze(0)).squeeze(0);
    this->_camera_center = this->_world_view_transform.inverse()[3].slice(0, 0, 3);
}

// TODO: I have skipped the resolution for now.
Camera loadCam(const ModelParameters& params, int id, CameraInfo& cam_info) {
    // Create a torch::Tensor from the image data
    torch::Tensor original_image_tensor = torch::from_blob(cam_info._img_data,
                                                           {cam_info._img_h, cam_info._img_w, cam_info._channels},        // img size
                                                           {cam_info._img_w * cam_info._channels, cam_info._channels, 1}, // stride
                                                           torch::kUInt8);
    original_image_tensor = original_image_tensor.to(torch::kFloat32).permute({2, 0, 1}).clone() / 255.f;

    free_image(cam_info._img_data); // we dont longer need the image here.
    cam_info._img_data = nullptr;   // Assure that we dont use the image data anymore.

    return Camera(cam_info._camera_ID, cam_info._R, cam_info._T, cam_info._fov_x, cam_info._fov_y, original_image_tensor,
                  cam_info._image_name, id);
}