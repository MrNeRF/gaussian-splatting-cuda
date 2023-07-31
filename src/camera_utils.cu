#include "camera_utils.cuh"
#include <eigen3/Eigen/Dense>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

Camera loadCam(CameraInfo& cam_info) {

    // ptyhon code
    //    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
    //                   FoVx=cam_info.FovX, FoVy=cam_info.FovY,
    //                   image=gt_image, gt_alpha_mask=loaded_mask,
    //                   image_name=cam_info.image_name, uid=id)

    return Camera(cam_info._camera_model);
}

void dump_JSON(const std::filesystem::path& file_path, const nlohmann::json& json_data) {
    std::ofstream file(file_path.string());
    if (file.is_open()) {
        file << json_data.dump(4); // Write the JSON data with indentation of 4 spaces
        file.close();
    } else {
        throw std::runtime_error("Could not open file " + file_path.string());
    }
}

// serialize camera to json
nlohmann::json camera_to_JSON(Camera cam) {

    Eigen::Matrix4d Rt = Eigen::Matrix4d::Zero();
    Rt.block<3, 3>(0, 0) = cam._R.transpose();
    Rt.block<3, 1>(0, 3) = cam._T;
    Rt(3, 3) = 1.0;

    Eigen::Matrix4d W2C = Rt.inverse();
    Eigen::Vector3d pos = W2C.block<3, 1>(0, 3);
    Eigen::Matrix3d rot = W2C.block<3, 3>(0, 0);
    std::vector<std::vector<double>> serializable_array_2d;
    for (int i = 0; i < rot.rows(); i++) {
        serializable_array_2d.push_back(std::vector<double>(rot.row(i).data(), rot.row(i).data() + rot.row(i).size()));
    }

    nlohmann::json camera_entry = {
        {"id", cam._camera_ID},
        {"img_name", cam._image_name},
        {"width", cam._width},
        {"height", cam._height},
        {"position", std::vector<double>(pos.data(), pos.data() + pos.size())},
        {"rotation", serializable_array_2d},
        {"fy", fov2focal(cam._fov_y, cam._height)},
        {"fx", fov2focal(cam._fov_x, cam._width)}};

    return camera_entry;
}