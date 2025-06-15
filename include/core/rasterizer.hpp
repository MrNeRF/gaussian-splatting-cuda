#pragma once

#include "Ops.h"
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include <torch/torch.h>

namespace gs {

    struct RenderOutput {
        torch::Tensor image;      // [..., C, H, W, channels]
        torch::Tensor alpha;      // [..., C, H, W, 1]
        torch::Tensor depth;      // [..., C, H, W, 1] - accumulated or expected depth
        torch::Tensor means2d;    // [..., N, 2]
        torch::Tensor depths;     // [..., N] - per-gaussian depths
        torch::Tensor radii;      // [..., N]
        torch::Tensor visibility; // [..., N]
        int width;
        int height;
    };

    enum class RenderMode {
        RGB,   // Color only
        D,     // Accumulated depth only
        ED,    // Expected depth only
        RGB_D, // Color + accumulated depth
        RGB_ED // Color + expected depth
    };

    // Wrapper function to use gsplat backend for rendering
    RenderOutput rasterize(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier = 1.0,
        bool packed = false,
        bool antialiased = false,
        RenderMode render_mode = RenderMode::RGB);

} // namespace gs