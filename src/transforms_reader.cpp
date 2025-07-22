#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <torch/torch.h>

#include "core/colmap_reader.hpp"
#include "core/transforms_reader.hpp"

namespace F = torch::nn::functional;

// Use std::numbers::pi instead of a custom PI constant.

float fov_deg_to_focal_length(int resolution, float fov_deg) {
    return 0.5f * (float)resolution / tanf(0.5f * fov_deg * std::numbers::pi / 180.0f);
}

float fov_rad_to_focal_length(int resolution, float fov_rad) {
    return 0.5f * (float)resolution / tanf(0.5f * fov_rad);
}

// Print the matrix nicely for debugging
void PrintMat(const torch::Tensor& transform_4x4) {
    std::cout << "4x4 Transformation Matrix:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < 4; ++j) {
            std::cout << std::fixed << std::setprecision(6) << std::setw(10)
                      << transform_4x4[i][j].item<float>();
            if (j < 3)
                std::cout << ", ";
        }
        std::cout << " ]" << std::endl;
    }
}

// Function to create a 3x3 rotation matrix around Y-axis embeded in 4x4 matrix
torch::Tensor createYRotationMatrix(float angle_radians) {
    torch::Tensor rotMat = torch::eye(4);
    float cos_angle = std::cos(angle_radians);
    float sin_angle = std::sin(angle_radians);

    // Rotation matrix around Y-axis by angle θ:
    // [cos(θ)   0   sin(θ) 0]
    // [  0      1     0    0]
    // [-sin(θ)  0   cos(θ) 0]
    // [0        0   0      1]

    rotMat[0][0] = cos_angle;  // cos(θ)
    rotMat[0][1] = 0.0f;       // 0
    rotMat[0][2] = sin_angle;  // sin(θ)
    rotMat[1][0] = 0.0f;       // 0
    rotMat[1][1] = 1.0f;       // 1
    rotMat[1][2] = 0.0f;       // 0
    rotMat[2][0] = -sin_angle; // -sin(θ)
    rotMat[2][1] = 0.0f;       // 0
    rotMat[2][2] = cos_angle;  // cos(θ)

    return rotMat;
}

std::tuple<std::vector<CameraData>, torch::Tensor> read_transforms_cameras_and_images(
    const std::filesystem::path& transPath) {

    std::filesystem::path transformsFile = transPath;
    if (std::filesystem::is_directory(transPath)) {
        if (std::filesystem::is_regular_file(transPath / "transforms_train.json")) {
            transformsFile = transPath / "transforms_train.json";
        } else if (std::filesystem::is_regular_file(transPath / "transforms.json")) {
            transformsFile = transPath / "transforms.json";
        } else {
            throw std::runtime_error("could not find transforms_traim.json nor transforms.json in " + transPath.string());
        }
    }

    if (!std::filesystem::is_regular_file(transformsFile)) {
        throw std::runtime_error(transformsFile.string() + " is not a valid file");
    }
    std::ifstream trans_file{transformsFile.string()};

    std::filesystem::path dir_path = transformsFile.parent_path();

    // should throw if parse fails
    nlohmann::json transforms = nlohmann::json::parse(trans_file, nullptr, true, true);
    int w = -1, h = -1;
    if (!transforms.contains("w") or !transforms.contains("h")) {
        throw std::runtime_error("transforms file must contain width (w) and height (h)");
    } else {
        w = int(transforms["w"]);
        h = int(transforms["h"]);
    }
    float fl_x = -1, fl_y = -1;
    if (transforms.contains("fl_x")) {
        fl_x = float(transforms["fl_x"]);
    } else if (transforms.contains("camera_angle_x")) {
        fl_x = fov_rad_to_focal_length(w, float(transforms["camera_angle_x"]));
    }

    if (transforms.contains("fl_y")) {
        fl_y = float(transforms["fl_y"]);
    } else if (transforms.contains("camera_angle_y")) {
        fl_y = fov_rad_to_focal_length(h, float(transforms["camera_angle_y"]));
    } else { // we should be  here in this scope only for blender - if w!=h then we must throw exception
        if (w != h) {
            throw std::runtime_error("no camera_angle_y expected w!=h");
        }
        fl_y = fl_x;
    }

    float cx = -1, cy = -1;
    if (transforms.contains("cx")) {
        cx = float(transforms["cx"]);
    } else {
        cx = 0.5 * w;
    }

    if (transforms.contains("cy")) {
        cy = float(transforms["cy"]);
    } else {
        cy = 0.5 * h;
    }

    uint64_t counter = 0;
    std::vector<CameraData> camerasdata;
    if (transforms.contains("frames") && transforms["frames"].is_array()) {
        for (int frameInd = 0; frameInd < transforms["frames"].size(); ++frameInd) {
            CameraData camdata;
            auto& frame = transforms["frames"][frameInd];
            if (!frame.contains("transform_matrix")) {
                throw std::runtime_error("expected all frames to contain transform_matrix");
            }

            nlohmann::json& jsonmatrix_start = frame["transform_matrix"];
            if (!(frame["transform_matrix"].is_array() and transforms["frames"].size() != 4)) {
                throw std::runtime_error("transform_matrix has the wrong dimensions");
            }

            // Create camera-to-world transform matrix
            torch::Tensor c2w = torch::empty({4, 4}, torch::kFloat32);

            // Fill the c2w matrix from the JSON data
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    c2w[i][j] = float(frame["transform_matrix"][i][j]);
                }
            }

            // Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            // c2w[:3, 1:3] *= -1
            c2w.slice(0, 0, 3).slice(1, 1, 3) *= -1;

            // Get the world-to-camera transform by computing inverse of c2w
            torch::Tensor w2c = torch::inverse(c2w);

            // fix so that the z direction will be the same (currently it is faceing downward)
            torch::Tensor fixMat = createYRotationMatrix(M_PI);
            w2c = torch::mm(w2c, fixMat);

            // Extract rotation matrix R (transposed due to 'glm' in CUDA code)
            // R = np.transpose(w2c[:3,:3])
            torch::Tensor R = w2c.slice(0, 0, 3).slice(1, 0, 3);

            // Extract translation vector T
            // T = w2c[:3, 3]
            torch::Tensor T = w2c.slice(0, 0, 3).slice(1, 3, 4).squeeze(1);

            auto image_path = dir_path / frame["file_path"];
            auto image_path_png = std::filesystem::path(image_path.string() + ".png");
            if (std::filesystem::exists(image_path_png)) {
                // blender data set has not extension, bust assumes png
                camdata._image_path = image_path_png;
            } else {
                camdata._image_path = image_path;
            }

            camdata._image_name = std::filesystem::path(camdata._image_path).filename().string();

            camdata._width = w;
            camdata._height = h;

            camdata._T = T;
            camdata._R = R;

            camdata._focal_x = fl_x;
            camdata._focal_y = fl_y;

            camdata._center_x = cx;
            camdata._center_y = cy;

            camdata._camera_model_type = gsplat::CameraModelType::PINHOLE;
            camdata._camera_ID = counter++;

            camerasdata.push_back(camdata);
        }
    }

    auto center = torch::zeros({1, 3}, torch::kFloat32);

    return {camerasdata, center};
}

PointCloud generate_random_point_cloud() {
    int numInitGaussian = 10000;

    uint64_t seed = 8128;
    // Set random seed for reproducibility
    torch::manual_seed(seed);

    torch::Tensor positions = torch::rand({numInitGaussian, 3}); // in [0, 1]
    positions = positions * 2.0 - 1.0;                           // now in [-1, 1]
    // Random RGB colors
    torch::Tensor colors = torch::randint(0, 256, {numInitGaussian, 3}, torch::kUInt8);

    return PointCloud(positions, colors);
}