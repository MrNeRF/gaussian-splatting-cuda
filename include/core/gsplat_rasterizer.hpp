#pragma once

#include <torch/torch.h>
#include "Ops.h"
#include "core/camera.hpp"
#include "core/splat_data.hpp"

namespace gs {

    struct GSplatRenderOutput {
        torch::Tensor image;         // rendered_image
        torch::Tensor means2d;       // 2D means
        torch::Tensor depths;        // depths
        torch::Tensor radii;         // per-Gaussian projected radius
        torch::Tensor camera_ids;    // for packed mode
        torch::Tensor gaussian_ids;  // for packed mode
    };

    // Wrapper function to use gsplat backend for rendering
    GSplatRenderOutput render_gsplat(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier = 1.0,
        bool packed = false);

} // namespace gs