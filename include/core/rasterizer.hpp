#pragma once

#include "Ops.h"
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include <torch/torch.h>

namespace gs {

    struct RenderOutput {
        torch::Tensor image;   // rendered_image
        torch::Tensor means2d; // 2D means
        torch::Tensor depths;  // depths
        torch::Tensor radii;   // per-Gaussian projected radius
        torch::Tensor visibility;

        int width;
        int height;
    };

    // Wrapper function to use gsplat backend for rendering
    RenderOutput rasterize(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier = 1.0,
        bool packed = false);

} // namespace gs