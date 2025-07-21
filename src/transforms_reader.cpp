#include <torch/torch.h>
#include <fstream>
#include <filesystem>
#include <numbers>
#include <nlohmann/json.hpp>

#include "core/colmap_reader.hpp"
#include "core/transforms_reader.hpp"

namespace F = torch::nn::functional;

const double PI = 3.141592653589793;

float fov_deg_to_focal_length(int resolution, float fov_deg) {
    return 0.5f * (float)resolution / tanf(0.5f * fov_deg * PI / 180.0f);
}

float fov_rad_to_focal_length(int resolution, float fov_rad) {
    return 0.5f * (float)resolution / tanf(0.5f * fov_rad);
}

std::tuple<std::vector<CameraData>, torch::Tensor> read_transforms_cameras_and_images(
    const std::filesystem::path& base,
    const std::string& images_folder) {

    std::vector<CameraData> camerasdata;

    std::ifstream trans_file{base.string()};
    nlohmann::json transforms = nlohmann::json::parse(trans_file, nullptr, true, true);

    if (!transforms.contains("w") or !transforms.contains("h"))
    {
        throw std::runtime_error("needs width and heigth in transforms file for now");
    }

    int w=-1, h = -1;


    if (transforms.contains("frames") && transforms["frames"].is_array()) {
        for (int numFrames = 0; numFrames < transforms["frames"].size(); ++numFrames) {
            CameraData camdata;
            auto& frame = transforms["frames"][numFrames];
            if (!frame.contains("transform_matrix") )
            {
                throw std::runtime_error("expected all frames to contain transform_matrix");
            }

            nlohmann::json& jsonmatrix_start = frame["transform_matrix"];
            if (! (frame["transform_matrix"].is_array() and transforms["frames"].size()!=4 ))
            {
                throw std::runtime_error("transform_matrix has the wrong dimensions");
            }
            torch::Tensor R = torch::empty({3, 3}, torch::kFloat32);
            for (int i=0;i<3;++i)
            {
                for (int j=0;j<3;++j)
                {
                    R[i][j] = float(frame["transform_matrix"][i][j]);
                }
            }

            torch::Tensor T = torch::zeros({3}, torch::kFloat32);
            for (int i=0;i<3;++i)
            {
                T[i] = float(frame["transform_matrix"][3][i]);
            }

            camdata._image_path = base / frame["file_path"];
            if (std::filesystem::exists(camdata._image_path /".png"))
            {
                // blender data set has not extension, bust assumes png
                camdata._image_path =camdata._image_path /".png";
            }

            camdata._image_name = std::filesystem::path(frame["frame"]).filename().string();

            camdata._width = int(transforms["w"]);
            camdata._height = int(transforms["h"]);

            camdata._R = R;
            camdata._T = T;

            if (transforms.contains("fl_x"))
            {
                camdata._focal_x = float(transforms["fl_x"]);
            }
            else if (transforms.contains("camera_angle_x")){
                camdata._focal_x = fov_rad_to_focal_length(camdata._width, float(transforms["camera_angle_x"]) );
            }

            //            camdata._focal_y = fx;
            //            camdata._center_x = out[i]._params[1].item<float>();
            //            camdata._center_y = out[i]._params[2].item<float>();
            camdata._camera_model_type = gsplat::CameraModelType::PINHOLE;

        }
    }



    return std::tuple<std::vector<CameraData>, torch::Tensor>();
}


PointCloud generate_random_point_cloud() {
    int numInitGaussian = 1000;

    uint64_t seed = 8128;
    // Set random seed for reproducibility
    torch::manual_seed(seed);

    torch::Tensor positions = torch::rand({numInitGaussian, 3}); // in [0, 1]
    positions = positions * 2.0 - 1.0;  // now in [-1, 1]
    // Random RGB colors
    torch::Tensor colors = torch::randint(0, 256, {numInitGaussian, 3}, torch::kUInt8);

    return PointCloud(positions, colors);
}