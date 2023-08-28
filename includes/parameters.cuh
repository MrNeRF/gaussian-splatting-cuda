// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include <filesystem>

struct OptimizationParameters {
    size_t iterations = 30'000;
    float position_lr_init = 0.00016f;
    float position_lr_final = 0.0000016f;
    float position_lr_delay_mult = 0.01f;
    int64_t position_lr_max_steps = 30'000;
    float feature_lr = 0.0025f;
    float percent_dense = 0.01f;
    float opacity_lr = 0.05f;
    float scaling_lr = 0.001f;
    float rotation_lr = 0.001f;
    float lambda_dssim = 0.2f;
    uint64_t densification_interval = 100;
    uint64_t opacity_reset_interval = 3'000;
    uint64_t densify_from_iter = 500;
    uint64_t densify_until_iter = 15'000;
    float densify_grad_threshold = 0.0002f;
    bool early_stopping = false;
    float convergence_threshold = 0.007f;
    bool empty_gpu_cache = false;
};

struct ModelParameters {
    int sh_degree = 3;
    std::filesystem::path source_path = "";
    std::filesystem::path output_path = "output";
    std::string images = "images";
    int resolution = -1;
    bool white_background = false;
    bool eval = false;
};