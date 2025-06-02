#include "core/gsplat_rasterizer.hpp"
#include "Ops.h"
#include <iostream>

namespace gs {

    GSplatRenderOutput render_gsplat(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier,
        bool packed) {

        // Ensure background tensor is on GPU
        bg_color = bg_color.to(torch::kCUDA);

        // Get camera parameters
        int width = viewpoint_camera.image_width();
        int height = viewpoint_camera.image_height();

        // Prepare viewmat and K
        torch::Tensor viewmat = viewpoint_camera.world_view_transform().unsqueeze(0); // [1, 4, 4]
        torch::Tensor K = torch::zeros({1, 3, 3}, torch::kCUDA);

        // Debug: Check viewmat shape
        std::cout << "Viewmat shape: " << viewmat.sizes() << std::endl;

        // Build intrinsic matrix from camera parameters
        float focal_x = width / (2.0f * std::tan(viewpoint_camera.FoVx() * 0.5f));
        float focal_y = height / (2.0f * std::tan(viewpoint_camera.FoVy() * 0.5f));

        K[0][0][0] = focal_x;
        K[0][1][1] = focal_y;
        K[0][0][2] = width / 2.0f;
        K[0][1][2] = height / 2.0f;
        K[0][2][2] = 1.0f;

        // Get Gaussian parameters
        torch::Tensor means3D = gaussian_model.get_xyz();
        torch::Tensor scales = gaussian_model.get_scaling();
        torch::Tensor rotations = gaussian_model.get_rotation();
        torch::Tensor opacities = gaussian_model.get_opacity();
        torch::Tensor shs = gaussian_model.get_features();

        // Check if we need gradients (during training)
        bool need_grads = means3D.requires_grad() || scales.requires_grad() ||
                          rotations.requires_grad() || opacities.requires_grad() ||
                          shs.requires_grad();

        // Ensure all tensors are on CUDA - use non_blocking for better performance
        auto cuda_options = torch::TensorOptions().device(torch::kCUDA);
        means3D = means3D.to(torch::kCUDA, /*non_blocking=*/true);
        scales = scales.to(torch::kCUDA, /*non_blocking=*/true);
        rotations = rotations.to(torch::kCUDA, /*non_blocking=*/true);
        opacities = opacities.to(torch::kCUDA, /*non_blocking=*/true);
        shs = shs.to(torch::kCUDA, /*non_blocking=*/true);

        // The .to() operation preserves gradient requirements, but let's verify
        if (need_grads) {
            std::cout << "Gradient requirements - means3D: " << means3D.requires_grad()
                      << ", scales: " << scales.requires_grad()
                      << ", rotations: " << rotations.requires_grad()
                      << ", opacities: " << opacities.requires_grad()
                      << ", shs: " << shs.requires_grad() << std::endl;
        }

        // Parameters for projection
        float eps2d = 0.3f;
        float near_plane = 0.01f;
        float far_plane = 100.0f;
        float radius_clip = 0.0f;
        bool calc_compensations = false;

        // Project Gaussians to 2D using gsplat
        torch::Tensor radii, means2d, depths, conics, compensations;
        torch::Tensor camera_ids, gaussian_ids, batch_ids;
        torch::Tensor indptr; // For packed mode

        if (packed) {
            // Use packed projection - the function returns 9 values:
            // (indptr, batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations)
            auto proj_results = gsplat::projection_ewa_3dgs_packed_fwd(
                means3D,
                torch::nullopt, // covars
                rotations,      // quats
                scales,
                opacities,
                viewmat,
                K,
                width,
                height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                calc_compensations,
                gsplat::CameraModelType::PINHOLE);

            // Unpack the 9 values returned (added batch_ids which was missing)
            std::tie(indptr, batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations) = proj_results;
        } else {
            // Use regular projection
            auto proj_results = gsplat::projection_ewa_3dgs_fused_fwd(
                means3D,
                torch::nullopt, // covars
                rotations,      // quats
                scales,
                opacities,
                viewmat,
                K,
                width,
                height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                calc_compensations,
                gsplat::CameraModelType::PINHOLE);

            std::tie(radii, means2d, depths, conics, compensations) = proj_results;
        }

        // Compute colors from spherical harmonics
        torch::Tensor colors;
        if (packed) {
            // For packed mode, we need to compute SH for visible Gaussians only
            // Extract camera position from the inverse of view matrix
            torch::Tensor viewmat_inv = torch::inverse(viewmat.squeeze(0));              // [4, 4]
            torch::Tensor campos = viewmat_inv.index({torch::indexing::Slice(0, 3), 3}); // [3] - last column, first 3 rows

            torch::Tensor dirs = means3D.index_select(0, gaussian_ids) - campos.unsqueeze(0); // [nnz, 3]
            torch::Tensor shs_visible = shs.index_select(0, gaussian_ids);                    // [nnz, K, 3]

            colors = gsplat::spherical_harmonics_fwd(
                gaussian_model.get_active_sh_degree(),
                dirs,
                shs_visible,
                torch::nullopt // masks
            );
        } else {
            // Regular SH computation
            // Extract camera position from the inverse of view matrix
            torch::Tensor viewmat_inv = torch::inverse(viewmat.squeeze(0));              // [4, 4]
            torch::Tensor campos = viewmat_inv.index({torch::indexing::Slice(0, 3), 3}); // [3] - last column, first 3 rows

            torch::Tensor dirs = means3D - campos.unsqueeze(0); // [N, 3]

            colors = gsplat::spherical_harmonics_fwd(
                gaussian_model.get_active_sh_degree(),
                dirs,
                shs,
                torch::nullopt // masks
            );
            colors = colors.unsqueeze(0); // [1, N, 3]
        }

        // Clamp colors (following Inria's implementation)
        colors = torch::clamp_min(colors + 0.5f, 0.0f);

        // Get tile information
        int tile_size = 16;
        int tile_width = (width + tile_size - 1) / tile_size;
        int tile_height = (height + tile_size - 1) / tile_size;

        // Intersect tiles
        torch::Tensor tiles_per_gauss, isect_ids, flatten_ids;
        std::tie(tiles_per_gauss, isect_ids, flatten_ids) = gsplat::intersect_tile(
            means2d,
            radii,
            depths,
            packed ? torch::optional<torch::Tensor>(camera_ids) : torch::nullopt,
            packed ? torch::optional<torch::Tensor>(gaussian_ids) : torch::nullopt,
            1, // C = 1 camera
            tile_size,
            tile_width,
            tile_height,
            true, // sort
            false // segmented
        );

        // Compute intersection offsets
        torch::Tensor isect_offsets = gsplat::intersect_offset(
            isect_ids,
            1, // C = 1 camera
            tile_width,
            tile_height);

        // Prepare opacities for rasterization
        torch::Tensor opacities_raster;
        if (packed) {
            opacities_raster = opacities.squeeze(-1).index_select(0, gaussian_ids);
        } else {
            opacities_raster = opacities.squeeze(-1).unsqueeze(0); // [1, N]
        }

        // Rasterize to pixels
        torch::Tensor render_colors, render_alphas, last_ids;
        std::tie(render_colors, render_alphas, last_ids) = gsplat::rasterize_to_pixels_3dgs_fwd(
            means2d,
            conics,
            colors,
            opacities_raster,
            bg_color.unsqueeze(0), // backgrounds [1, 3]
            torch::nullopt,        // masks
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids);

        // Extract the rendered image for single camera
        render_colors = render_colors[0]; // [H, W, 3]

        // Transpose to match expected format [3, H, W]
        render_colors = render_colors.permute({2, 0, 1});

        // Debug gradient tracking
        if (need_grads && !render_colors.requires_grad()) {
            std::cerr << "WARNING: Rendered image does not have gradients enabled!" << std::endl;
            std::cerr << "This will cause backward() to fail during training." << std::endl;
        }

        GSplatRenderOutput output;
        output.image = render_colors;
        output.means2d = means2d;
        output.depths = depths;
        output.radii = radii;
        output.camera_ids = camera_ids;
        output.gaussian_ids = gaussian_ids;

        return output;
    }

} // namespace gs