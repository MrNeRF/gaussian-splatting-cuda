// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include <expected>
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
            bool skip_intermediate_saving = false;            // Skip saving intermediate results and only save final output
            bool enable_eval = false;                         // Only evaluate when explicitly enabled
            bool enable_save_eval_images = true;              // Save during evaluation images
            bool headless = false;                            // Disable visualization during training
            std::string render_mode = "RGB";                  // Render mode: RGB, D, ED, RGB_D, RGB_ED
            std::string strategy = "mcmc";                    // Optimization strategy: mcmc, default.

            // Bilateral grid parameters
            bool use_bilateral_grid = false;
            int bilateral_grid_X = 16;
            int bilateral_grid_Y = 16;
            int bilateral_grid_W = 8;
            float bilateral_grid_lr = 2e-3f;
            float tv_loss_weight = 10.f;

            // Default strategy specific parameters
            float prune_opacity = 0.005f;
            float grow_scale3d = 0.01f;
            float grow_scale2d = 0.05f;
            float prune_scale3d = 0.1f;
            float prune_scale2d = 0.15f;
            size_t reset_every = 3'000;
            size_t pause_refine_after_reset = 0;
            bool revised_opacity = false;

            float steps_scaler = 0.f;  // If < 0, step size scaling is disabled
            bool antialiasing = false; // Enable antialiasing in rendering
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

            // Viewer mode specific
            std::filesystem::path ply_path = "";
        };

        // Modern C++23 functions returning expected values
        std::expected<OptimizationParameters, std::string> read_optim_params_from_json(const std::string strategy);

        // Save training parameters to JSON
        std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path);
    } // namespace param
} // namespace gs
