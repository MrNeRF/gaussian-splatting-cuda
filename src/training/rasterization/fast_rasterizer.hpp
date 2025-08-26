#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "rasterization_api.h"
#include "rasterizer.hpp"

namespace gs {

    // Wrapper function to use fastgs backend for rendering
    RenderOutput fast_rasterize(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color);

} // namespace gs