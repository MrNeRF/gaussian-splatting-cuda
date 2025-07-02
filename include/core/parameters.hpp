// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace gs {
    namespace param {
        struct OptimizationParameters {
            size_t iterations = 30'000;
            size_t sh_degree_interval = 1'000;
            float means_lr = 0.00016f;
            float shs_lr = 0.0025f;
            float opacity_lr = 0.05f;
            float scaling_lr = 0.005f;
            float rotation_lr = 0.001f;
            float lambda_dssim = 0.2f;
            float min_opacity = 0.005f;
            size_t refine_every = 100;
            size_t start_refine = 500;
            size_t stop_refine = 25'000;
            float grad_threshold = 0.0002f;
            int sh_degree = 3;
            float opacity_reg = 0.01f;
            float scale_reg = 0.01f;
            float init_opacity = 0.5f;
            float init_scaling = 0.1f;
            int max_cap = 1000000;
            std::vector<size_t> eval_steps = {7'000, 30'000}; // Steps to evaluate the model
            std::vector<size_t> save_steps = {7'000, 30'000}; // Steps to save the model
            bool enable_eval = false;                         // Only evaluate when explicitly enabled
            bool enable_save_eval_images = false;             // Save during evaluation images
            bool enable_viz = false;                          // Enable visualization during training
            std::string render_mode = "RGB";                  // Render mode: RGB, D, ED, RGB_D, RGB_ED

            // Bilateral grid parameters
            bool use_bilateral_grid = false;
            int bilateral_grid_X = 16;
            int bilateral_grid_Y = 16;
            int bilateral_grid_W = 8;
            float bilateral_grid_lr = 2e-3;
            float tv_loss_weight = 10.0f;

            int steps_scaler = 1;
            bool selective_adam = false; // Use Selective Adam optimizer
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

        // Save training parameters to JSON
        void save_training_parameters_to_json(const TrainingParameters& params,
                                              const std::filesystem::path& output_path);
    } // namespace param
} // namespace gs