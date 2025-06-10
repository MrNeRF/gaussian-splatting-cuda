#include "core/rasterizer.hpp"
#include "Ops.h"
#include "core/rasterizer_autograd.hpp"
#include <torch/torch.h>

namespace gs {

    using namespace torch::indexing;

    // Main render function
    RenderOutput rasterize(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier,
        bool packed) {

        // Ensure we don't use packed mode (not supported in this implementation)
        TORCH_CHECK(!packed, "Packed mode is not supported in this implementation");

        // Get camera parameters
        const int image_height = static_cast<int>(viewpoint_camera.image_height());
        const int image_width = static_cast<int>(viewpoint_camera.image_width());

        // Prepare viewmat and K
        auto viewmat = viewpoint_camera.world_view_transform().to(torch::kCUDA);
        TORCH_CHECK(viewmat.dim() == 3 && viewmat.size(0) == 1 && viewmat.size(1) == 4 && viewmat.size(2) == 4,
                    "viewmat must be [1, 4, 4] after transpose and unsqueeze, got ", viewmat.sizes());
        TORCH_CHECK(viewmat.is_cuda(), "viewmat must be on CUDA");

        const auto K = viewpoint_camera.K().to(torch::kCUDA);
        TORCH_CHECK(K.is_cuda(), "K must be on CUDA");

        // Get Gaussian parameters
        auto means3D = gaussian_model.get_means();
        auto opacities = gaussian_model.get_opacity();
        if (opacities.dim() == 2 && opacities.size(1) == 1) {
            opacities = opacities.squeeze(-1);
        }
        const auto scales = gaussian_model.get_scaling();
        const auto rotations = gaussian_model.get_rotation();
        const auto sh_coeffs = gaussian_model.get_shs();
        const int sh_degree = gaussian_model.get_active_sh_degree();

        // Validate Gaussian parameters
        const int N = static_cast<int>(means3D.size(0));
        TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3,
                    "means3D must be [N, 3], got ", means3D.sizes());
        TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == N,
                    "opacities must be [N], got ", opacities.sizes());
        TORCH_CHECK(scales.dim() == 2 && scales.size(0) == N && scales.size(1) == 3,
                    "scales must be [N, 3], got ", scales.sizes());
        TORCH_CHECK(rotations.dim() == 2 && rotations.size(0) == N && rotations.size(1) == 4,
                    "rotations must be [N, 4], got ", rotations.sizes());
        TORCH_CHECK(sh_coeffs.dim() == 3 && sh_coeffs.size(0) == N && sh_coeffs.size(2) == 3,
                    "sh_coeffs must be [N, K, 3], got ", sh_coeffs.sizes());

        // Device checks for Gaussian parameters
        TORCH_CHECK(means3D.is_cuda(), "means3D must be on CUDA");
        TORCH_CHECK(opacities.is_cuda(), "opacities must be on CUDA");
        TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA");
        TORCH_CHECK(rotations.is_cuda(), "rotations must be on CUDA");
        TORCH_CHECK(sh_coeffs.is_cuda(), "sh_coeffs must be on CUDA");

        // Ensure background color is properly shaped and on CUDA
        if (!bg_color.defined() || bg_color.numel() == 0) {
            bg_color = torch::zeros({1, 3}, means3D.options());
        } else {
            bg_color = bg_color.view({1, -1}).to(torch::kCUDA);
            TORCH_CHECK(bg_color.size(0) == 1 && bg_color.size(1) == 3,
                        "bg_color must be reshapeable to [1, 3], got ", bg_color.sizes());
        }
        TORCH_CHECK(bg_color.is_cuda(), "bg_color must be on CUDA");

        const float eps2d = 0.3f;
        const float near_plane = 0.01f;
        const float far_plane = 10000.0f;
        const float radius_clip = 0.0f;
        const int tile_size = 16;

        // Step 1: Projection
        auto proj_settings = torch::tensor({(float)image_width,
                                            (float)image_height,
                                            eps2d,
                                            near_plane,
                                            far_plane,
                                            radius_clip,
                                            scaling_modifier},
                                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto proj_outputs = ProjectionFunction::apply(
            means3D, rotations, scales, opacities, viewmat, K, proj_settings);

        auto radii = proj_outputs[0];
        auto means2d = proj_outputs[1];
        auto depths = proj_outputs[2];
        auto conics = proj_outputs[3];
        auto compensations = proj_outputs[4];

        // Create means2d with gradient tracking for backward compatibility
        auto means2d_with_grad = means2d.squeeze(0).contiguous();
        means2d_with_grad.set_requires_grad(true);
        means2d_with_grad.retain_grad();

        // Step 2: Compute colors from SH
        auto sh_degree_tensor = torch::tensor({sh_degree}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        auto color_outputs = SphericalHarmonicsFunction::apply(
            sh_coeffs, means3D, viewmat, radii, sh_degree_tensor);
        auto colors = color_outputs[0];

        // Apply the SH offset and clamping for rendering (shift from [-0.5, 0.5] to [0, 1])
        colors = torch::clamp_min(colors + 0.5f, 0.0f);

        // Step 3: Apply opacity with compensations
        auto final_opacities = opacities.unsqueeze(0) * compensations;
        TORCH_CHECK(final_opacities.is_cuda(), "final_opacities must be on CUDA");

        // Step 4: Tile intersection (no autograd needed)
        const int tile_width = (image_width + tile_size - 1) / tile_size;
        const int tile_height = (image_height + tile_size - 1) / tile_size;

        const auto isect_results = gsplat::intersect_tile(
            means2d, radii, depths, {}, {},
            1, tile_size, tile_width, tile_height,
            true);

        const auto tiles_per_gauss = std::get<0>(isect_results);
        const auto isect_ids = std::get<1>(isect_results);
        const auto flatten_ids = std::get<2>(isect_results);

        auto isect_offsets = gsplat::intersect_offset(
            isect_ids, 1, tile_width, tile_height);
        isect_offsets = isect_offsets.reshape({1, tile_height, tile_width});

        TORCH_CHECK(tiles_per_gauss.is_cuda(), "tiles_per_gauss must be on CUDA");
        TORCH_CHECK(isect_ids.is_cuda(), "isect_ids must be on CUDA");
        TORCH_CHECK(flatten_ids.is_cuda(), "flatten_ids must be on CUDA");
        TORCH_CHECK(isect_offsets.is_cuda(), "isect_offsets must be on CUDA");

        // Step 5: Rasterization
        auto raster_settings = torch::tensor({(float)image_width,
                                              (float)image_height,
                                              (float)tile_size},
                                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto raster_outputs = RasterizationFunction::apply(
            means2d, conics, colors, final_opacities, bg_color,
            isect_offsets, flatten_ids, raster_settings);

        auto rendered_image = raster_outputs[0];
        auto rendered_alpha = raster_outputs[1];

        // Prepare output
        RenderOutput result;
        result.image = torch::clamp_max(rendered_image.squeeze(0).permute({2, 0, 1}), 1.0f);
        result.means2d = means2d_with_grad;
        result.depths = depths.squeeze(0);
        result.radii = std::get<0>(radii.squeeze(0).max(-1));
        result.visibility = (result.radii > 0);
        result.width = image_width;
        result.height = image_height;

        // Final device checks for outputs
        TORCH_CHECK(result.image.is_cuda(), "result.image must be on CUDA");
        TORCH_CHECK(result.means2d.is_cuda(), "result.means2d must be on CUDA");
        TORCH_CHECK(result.depths.is_cuda(), "result.depths must be on CUDA");
        TORCH_CHECK(result.radii.is_cuda(), "result.radii must be on CUDA");
        TORCH_CHECK(result.visibility.is_cuda(), "result.visibility must be on CUDA");

        return result;
    }

} // namespace gs