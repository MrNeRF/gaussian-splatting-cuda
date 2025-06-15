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

    // Helper function to check if render mode includes depth
    static inline bool renderModeHasDepth(RenderMode mode) {
        return mode != RenderMode::RGB;
    }

    // Helper function to check if render mode includes RGB
    static inline bool renderModeHasRGB(RenderMode mode) {
        return mode == RenderMode::RGB ||
               mode == RenderMode::RGB_D ||
               mode == RenderMode::RGB_ED;
    }

    static inline RenderMode stringToRenderMode(const std::string& mode) {
        if (mode == "RGB")
            return RenderMode::RGB;
        else if (mode == "D")
            return RenderMode::D;
        else if (mode == "ED")
            return RenderMode::ED;
        else if (mode == "RGB_D")
            return RenderMode::RGB_D;
        else if (mode == "RGB_ED")
            return RenderMode::RGB_ED;
        else
            throw std::runtime_error("Invalid render mode: " + mode);
    }

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