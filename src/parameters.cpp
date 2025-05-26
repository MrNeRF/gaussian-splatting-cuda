// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#include "core/parameters.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace gs {
    namespace param {
        OptimizationParameters read_optim_params_from_json() {

            // automatically get the root path. Works only on linux, I guess.
            std::filesystem::path executablePath = std::filesystem::canonical("/proc/self/exe");
            std::filesystem::path parentDir = executablePath.parent_path().parent_path();
            std::filesystem::path json_path = parentDir / "parameter/optimization_params.json";
            // Check if the file exists before trying to open it
            if (!std::filesystem::exists(json_path)) {
                throw std::runtime_error("Error: " + json_path.string() + " does not exist!");
            }

            std::ifstream file(json_path);
            if (!file.is_open()) {
                throw std::runtime_error("OptimizationParameter file could not be opened.");
            }

            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string jsonString = buffer.str();
            file.close(); // Explicitly close the file

            // Parse the JSON string
            nlohmann::json json = nlohmann::json::parse(jsonString);

            OptimizationParameters params;
            params.iterations = json["iterations"];
            params.position_lr_init = json["position_lr_init"];
            params.position_lr_final = json["position_lr_final"];
            params.position_lr_delay_mult = json["position_lr_delay_mult"];
            params.position_lr_max_steps = json["position_lr_max_steps"];
            params.feature_lr = json["feature_lr"];
            params.percent_dense = json["percent_dense"];
            params.opacity_lr = json["opacity_lr"];
            params.scaling_lr = json["scaling_lr"];
            params.rotation_lr = json["rotation_lr"];
            params.lambda_dssim = json["lambda_dssim"];
            params.min_opacity = json["min_opacity"];
            params.densification_interval = json["densification_interval"];
            params.opacity_reset_interval = json["opacity_reset_interval"];
            params.densify_from_iter = json["densify_from_iter"];
            params.densify_until_iter = json["densify_until_iter"];
            params.densify_grad_threshold = json["densify_grad_threshold"];

            return params;
        }
    } // namespace param
} // namespace gs
