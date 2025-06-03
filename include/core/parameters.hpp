// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include <filesystem>

namespace gs {
    namespace param {
        struct OptimizationParameters {
            size_t iterations = 30'000;
            float position_lr_init = 0.00016f;
            float feature_lr = 0.0025f;
            float opacity_lr = 0.05f;
            float scaling_lr = 0.001f;
            float rotation_lr = 0.001f;
            float lambda_dssim = 0.2f;
            float min_opacity = 0.005f;
            size_t densification_interval = 100;
            size_t opacity_reset_interval = 3'000;
            size_t densify_from_iter = 500;
            size_t densify_until_iter = 15'000;
            float densify_grad_threshold = 0.0002f;
            int sh_degree = 3;
            float opacity_reg = 0.01f;
            float scale_reg = 0.01f;
        };

        struct DatasetConfig {
            std::filesystem::path data_path = "";
            std::filesystem::path output_path = "output";
            std::string images = "images";
            int resolution = -1;
        };

        struct TrainingParameters {
            DatasetConfig dataset;
            OptimizationParameters optimization;
        };

        OptimizationParameters read_optim_params_from_json();
    } // namespace param
} // namespace gs