#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include <tuple>

namespace gs::rendering {

    torch::Tensor rasterize(
        Camera& viewpoint_camera,
        SplatData& gaussian_model,
        torch::Tensor& bg_color);

} // namespace gs::rendering