#pragma once

#include "core/camera.hpp"
#include "core/gsplat_rasterizer.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include <cmath>
#include <torch/torch.h>

struct RenderOutput {
    torch::Tensor image;         // rendered_image
    torch::Tensor viewspace_pts; // means2D
    torch::Tensor visibility;    // radii > 0
    torch::Tensor radii;         // per-Gaussian projected radius
    int width;
    int height;
    int n_cameras = 1;
};

inline RenderOutput render_with_gsplat(Camera& viewpoint_camera,
                                       const SplatData& gaussian_model,
                                       torch::Tensor& bg_color,
                                       float scaling_modifier = 1.0) {
    // Use gsplat backend with proper namespace
    auto gsplat_output = gs::render_gsplat(viewpoint_camera, gaussian_model, bg_color, scaling_modifier, false);

    // Convert to RenderOutput format
    RenderOutput output;
    output.image = gsplat_output.image;

    // IMPORTANT: Copy the tensor and ensure gradient retention
    output.viewspace_pts = gsplat_output.means2d;

    // GSplat radii is [N, 2], need to check both dimensions
    output.visibility = (gsplat_output.radii > 0).any(-1); // any(-1) reduces [N, 2] to [N]
    output.radii = std::get<0>(gsplat_output.radii.max(-1));

    // Set dimensions
    output.width = viewpoint_camera.image_width();
    output.height = viewpoint_camera.image_height();
    output.n_cameras = 1;

    return output;
}
