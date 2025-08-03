#pragma once

#include "core/rasterizer.hpp"
#include "Ops.h"
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include <torch/torch.h>

namespace tgs {

    gs::RenderOutput rasterize(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier/*=1*/,
        bool packed/*=false*/,
        bool antialiased/*=false*/,
        gs::RenderMode render_mode);

}