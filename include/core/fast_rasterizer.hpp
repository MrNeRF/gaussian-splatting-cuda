#pragma once

#include "rasterization_api.h"
#include "core/rasterizer.hpp"
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include <torch/torch.h>

namespace gs {

    // Wrapper function to use fastgs backend for rendering
    RenderOutput fast_rasterize(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color);

} // namespace gs