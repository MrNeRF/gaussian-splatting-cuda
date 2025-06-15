#pragma once

#include "Ops.h"
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include <torch/torch.h>

namespace gs {

    // Autograd function for projection
    class ProjectionFunction : public torch::autograd::Function<ProjectionFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor means3D,   // [N, 3]
            torch::Tensor quats,     // [N, 4]
            torch::Tensor scales,    // [N, 3]
            torch::Tensor opacities, // [N]
            torch::Tensor viewmat,   // [C, 4, 4]
            torch::Tensor K,         // [C, 3, 3]
            torch::Tensor settings); // [7] tensor containing projection settings

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);
    };

    // Autograd function for spherical harmonics
    class SphericalHarmonicsFunction : public torch::autograd::Function<SphericalHarmonicsFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor sh_degree_tensor, // [1] containing sh_degree
            torch::Tensor dirs,             // [N, 3]
            torch::Tensor coeffs,           // [N, K, 3]
            torch::Tensor masks);           // [N] optional boolean masks

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);
    };

    // Autograd function for rasterization
    class RasterizationFunction : public torch::autograd::Function<RasterizationFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor means2d,       // [C, N, 2]
            torch::Tensor conics,        // [C, N, 3]
            torch::Tensor colors,        // [C, N, channels] - may include depth
            torch::Tensor opacities,     // [C, N]
            torch::Tensor bg_color,      // [C, channels] - may include depth
            torch::Tensor isect_offsets, // [C, tile_height, tile_width]
            torch::Tensor flatten_ids,   // [nnz]
            torch::Tensor settings);     // [3] containing width, height, tile_size

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);
    };

    // Autograd function for quat_scale_to_covar_preci - shared between rasterizer and tests
    class QuatScaleToCovarPreciFunction : public torch::autograd::Function<QuatScaleToCovarPreciFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor quats,
            torch::Tensor scales,
            torch::Tensor settings); // [3] tensor containing [compute_covar, compute_preci, triu]

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);
    };

} // namespace gs