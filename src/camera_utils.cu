#include "camera_utils.cuh"
#include <nlohmann/json.hpp>

nlohmann::json camera_to_JSON(Camera cam) {

    // Eigen::Matrix4d Rt = Eigen::Matrix4d::Zero();
    // Rt.block<3,3>(0,0) = camera._R.transpose();
    // Rt.block<3,1>(0,3) = camera._T;
    // Rt(3,3) = 1.0;

    // Eigen::Matrix4d W2C = Rt.inverse();
    // Eigen::Vector3d pos = W2C.block<3,1>(0,3);
    // Eigen::Matrix3d rot = W2C.block<3,3>(0,0);
    // std::vector<std::vector<double>> serializable_array_2d;
    // for (int i = 0; i < rot.rows(); i++) {
    // serializable_array_2d.push_back(std::vector<double>(rot.row(i).data(), rot.row(i).data() + rot.row(i).size()));
    //}

    nlohmann::json camera_entry = {
        {"id", cam._camera_ID}
        //{"img_name", cam._image_name},
        //{"width", cam._image_width},
        //{"height", cam._image_height},
        //{"position", std::vector<double>(cam._position.data(), cam._position.data() + cam._position.size())},
        //{"rotation", std::vector<double>(cam._rotation.data(), cam._rotation.data() + cam._rotation.size())},
        //{"fy", fov2focal(cam._fov_y, cam._image_height)},
        //{"fx", fov2focal(cam._fov_x, cam._image_width)}
        //{"img_name", camera._image_name},
        //{"width", camera._image_width},
        //{"height", camera._image_height},
        //{"position", std::vector<double>(pos.data(), pos.data() + pos.size())},
        //{"rotation", serializable_array_2d},
        //{"fy", fov2focal(camera._fov_y, camera._image_height)},
        //{"fx", fov2focal(camera._fov_x, camera._image_width)}
    };

    return camera_entry;
}