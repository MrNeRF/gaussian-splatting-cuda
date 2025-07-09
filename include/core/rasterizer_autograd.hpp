#pragma once

#include "Ops.h"
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include <torch/torch.h>

namespace gs {
    struct ProjectionSettings {
        int width;
        int height;
        float eps2d;
        float near_plane;
        float far_plane;
        float radius_clip;
        float scaling_modifier;
    };

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
            ProjectionSettings settings);

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

    struct RasterizationSettings {
        int width;
        int height;
        int tile_size;
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
            RasterizationSettings settings);

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);
    };

    struct QuatScaleToCovarPreciSettings {
        bool compute_covar;
        bool compute_preci;
        bool triu;
    };

    // Autograd function for quat_scale_to_covar_preci - shared between rasterizer and tests
    class QuatScaleToCovarPreciFunction : public torch::autograd::Function<QuatScaleToCovarPreciFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor quats,
            torch::Tensor scales,
            QuatScaleToCovarPreciSettings settings);

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);
    };

    // 3DGUT Functions
    struct GUTProjectionSettings {
        int width;
        int height;
        float eps2d;
        float near_plane;
        float far_plane;
        float radius_clip;
        float scaling_modifier;
        gsplat::CameraModelType camera_model;
    };

    // Autograd function for projection
    torch::autograd::tensor_list fully_fused_projection_with_ut(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor means3D,   // [N, 3]
            torch::Tensor quats,     // [N, 4]
            torch::Tensor scales,    // [N, 3]
            torch::Tensor opacities, // [N]
            torch::Tensor viewmat,   // [C, 4, 4]
            torch::Tensor K,         // [C, 3, 3]
            std::optional<torch::Tensor> radial_coeffs, // [..., C, 6] or [..., C, 4]
            std::optional<torch::Tensor> tangential_coeffs, // [..., C, 2]
            std::optional<torch::Tensor> thin_prism_coeffs, // [..., C, 4]
            GUTProjectionSettings settings,
            UnscentedTransformParameters ut_params);

    struct GUTRasterizationSettings {
        int width;
        int height;
        int tile_size;
        gsplat::CameraModelType camera_model;
    };

    // Autograd function for rasterization
    class GUTRasterizationFunction : public torch::autograd::Function<GUTRasterizationFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor means3D,   // [N, 3]
             torch::Tensor quats,     // [N, 4]
             torch::Tensor scales,    // [N, 3]
             torch::Tensor colors,        // [N, C]
             torch::Tensor opacities, // [N]
             torch::Tensor bg_color,      // [N, C]
             torch::Tensor masks, // [N, C, tile_height, tile_width]
             torch::Tensor viewmat,   // [C, 4, 4]
             torch::Tensor K,         // [C, 3, 3]
             torch::Tensor radial_coeffs, // [..., C, 6] or [..., C, 4]
             torch::Tensor tangential_coeffs, // [..., C, 2]
             torch::Tensor thin_prism_coeffs, // [..., C, 4]
             torch::Tensor isect_offsets, // [C, tile_height, tile_width]
                torch::Tensor flatten_ids,   // [nnz]
            GUTRasterizationSettings settings,
            UnscentedTransformParameters ut_params);


        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);
    };
} // namespace gs