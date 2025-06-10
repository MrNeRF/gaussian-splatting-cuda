// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include <filesystem>
#include <vector>

namespace gs {
    namespace param {
        struct OptimizationParameters {
            size_t iterations = 30'000;
            float means_lr = 0.00016f;
            float shs_lr = 0.0025f;
            float opacity_lr = 0.05f;
            float scaling_lr = 0.005f;
            float rotation_lr = 0.001f;
            float lambda_dssim = 0.2f;
            float min_opacity = 0.005f;
            size_t growth_interval = 100;
            size_t start_densify = 500;
            size_t stop_densify = 15'000;
            float grad_threshold = 0.0002f;
            int sh_degree = 3;
            float opacity_reg = 0.01f;
            float scale_reg = 0.01f;
            int max_cap = 1000000;
            std::vector<size_t> eval_steps = {7000, 30000}; // Steps to evaluate the model
            std::vector<size_t> save_steps = {7000, 30000}; // Steps to save the model
            bool enable_eval = false;                       // Only evaluate when explicitly enabled
        };

        struct DatasetConfig {
            std::filesystem::path data_path = "";
            std::filesystem::path output_path = "output";
            std::string images = "images";
            int resolution = -1;
            int test_every = 8;
        };

        struct TrainingParameters {
            DatasetConfig dataset;
            OptimizationParameters optimization;
        };

        OptimizationParameters read_optim_params_from_json();
    } // namespace param
} // namespace gs