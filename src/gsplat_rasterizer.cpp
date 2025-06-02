#include "core/gsplat_rasterizer.hpp"
#include "Ops.h"
#include "core/debug_utils.hpp"
#include <cmath>
#include <torch/torch.h>

namespace gs {

    using namespace torch::indexing;

    // Custom autograd function for gsplat rasterization
    class _GSplatRasterizeGaussians : public torch::autograd::Function<_GSplatRasterizeGaussians> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor means3D,
            torch::Tensor quats,
            torch::Tensor scales,
            torch::Tensor opacities,
            torch::Tensor sh_coeffs,
            torch::Tensor viewmats,
            torch::Tensor Ks,
            torch::Tensor bg,
            int image_height,
            int image_width,
            float eps2d,
            float near_plane,
            float far_plane,
            float radius_clip,
            int sh_degree,
            bool calc_compensations,
            int camera_model_int) {

            // Convert camera model int to enum
            auto camera_model = static_cast<gsplat::CameraModelType>(camera_model_int);

            // Get dimensions
            int N = means3D.size(0);
            int C = viewmats.size(0);

            // Ensure all inputs are on CUDA and contiguous
            means3D = means3D.to(torch::kCUDA).contiguous();
            quats = quats.to(torch::kCUDA).contiguous();
            scales = scales.to(torch::kCUDA).contiguous();
            opacities = opacities.to(torch::kCUDA).contiguous();
            sh_coeffs = sh_coeffs.to(torch::kCUDA).contiguous();
            viewmats = viewmats.to(torch::kCUDA).contiguous();
            Ks = Ks.to(torch::kCUDA).contiguous();
            if (bg.defined()) {
                bg = bg.to(torch::kCUDA).contiguous();
            }

            // Step 1: Projection using gsplat's fully fused projection
            auto proj_results = gsplat::projection_ewa_3dgs_fused_fwd(
                means3D,
                {}, // covars (optional, we use quats/scales)
                quats,
                scales,
                opacities,
                viewmats,
                Ks,
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                calc_compensations,
                camera_model);

            auto radii = std::get<0>(proj_results).contiguous();
            auto means2d = std::get<1>(proj_results).contiguous();
            auto depths = std::get<2>(proj_results).contiguous();
            auto conics = std::get<3>(proj_results).contiguous();
            auto compensations = std::get<4>(proj_results);

            // Ensure compensations is always a valid tensor
            if (!compensations.defined()) {
                compensations = torch::ones({C, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            } else {
                compensations = compensations.contiguous();
            }

            // Apply compensations to opacities if calculated
            torch::Tensor final_opacities = opacities.unsqueeze(0).expand({C, N}).to(torch::kCUDA).contiguous();
            if (calc_compensations) {
                final_opacities = final_opacities * compensations;
            }

            // Step 2: Compute colors from spherical harmonics
            torch::Tensor colors;
            if (sh_degree > 0 && sh_coeffs.size(1) > 1) {
                // Compute camera positions
                auto viewmats_inv = torch::inverse(viewmats);                   // [C, 4, 4]
                auto campos = viewmats_inv.index({Slice(), Slice(None, 3), 3}); // [C, 3]

                // Compute view directions
                auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1); // [C, N, 3]

                // Create masks based on radii visibility
                auto masks = (radii > 0).all(-1); // [C, N]

                // Compute SH for each camera
                std::vector<torch::Tensor> color_list;
                for (int c = 0; c < C; ++c) {
                    auto cam_dirs = dirs[c];  // [N, 3]
                    auto cam_mask = masks[c]; // [N]

                    auto cam_colors = gsplat::spherical_harmonics_fwd(
                        sh_degree,
                        cam_dirs,
                        sh_coeffs,
                        cam_mask); // [N, 3]

                    color_list.push_back(cam_colors);
                }
                colors = torch::stack(color_list, 0); // [C, N, 3]

                // Apply clamping as in Python
                colors = torch::clamp_min(colors + 0.5f, 0.0f);
            } else {
                // No SH, just use DC component
                colors = sh_coeffs.index({Slice(), 0, Slice()}); // [N, 3]
                colors = colors.unsqueeze(0).expand({C, -1, -1}).contiguous();
                colors = torch::clamp_min(colors + 0.5f, 0.0f);
            }

            // Step 3: Tile intersection
            int tile_size = 16;
            int tile_width = (image_width + tile_size - 1) / tile_size;
            int tile_height = (image_height + tile_size - 1) / tile_size;

            auto isect_results = gsplat::intersect_tile(
                means2d,
                radii,
                depths,
                {}, // image_ids (optional for packed mode)
                {}, // gaussian_ids (optional for packed mode)
                C,  // number of images
                tile_size,
                tile_width,
                tile_height,
                true, // sort
                false // segmented
            );

            auto tiles_per_gauss = std::get<0>(isect_results);
            auto isect_ids = std::get<1>(isect_results);
            auto flatten_ids = std::get<2>(isect_results);

            // Step 4: Encode offsets
            auto isect_offsets = gsplat::intersect_offset(
                isect_ids,
                C,
                tile_width,
                tile_height);

            // Step 5: Rasterize to pixels
            auto raster_results = gsplat::rasterize_to_pixels_3dgs_fwd(
                means2d,
                conics,
                colors,
                final_opacities,
                bg.defined() ? bg : torch::Tensor(),
                {}, // masks (optional)
                image_width,
                image_height,
                tile_size,
                isect_offsets,
                flatten_ids);

            auto rendered_image = std::get<0>(raster_results).contiguous();
            auto rendered_alpha = std::get<1>(raster_results).contiguous();
            auto last_ids = std::get<2>(raster_results).contiguous();

            // Convert alpha from double to float
            rendered_alpha = rendered_alpha.to(torch::kFloat32);

            // Save for backward
            ctx->save_for_backward({means3D, quats, scales, sh_coeffs, viewmats, Ks,
                                    radii, means2d, depths, conics,
                                    compensations,
                                    final_opacities,
                                    colors,
                                    isect_offsets, flatten_ids, rendered_alpha, last_ids});

            ctx->saved_data["image_width"] = image_width;
            ctx->saved_data["image_height"] = image_height;
            ctx->saved_data["tile_size"] = tile_size;
            ctx->saved_data["eps2d"] = eps2d;
            ctx->saved_data["camera_model"] = camera_model_int;
            ctx->saved_data["calc_compensations"] = calc_compensations;
            ctx->saved_data["bg"] = bg;
            ctx->saved_data["sh_degree"] = sh_degree;

            return {rendered_image, rendered_alpha, radii, means2d, depths};
        }

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {

            // Ensure gradients are on CUDA
            auto grad_rendered_image = grad_outputs[0].defined() ? grad_outputs[0].to(torch::kCUDA).contiguous() : torch::Tensor();
            auto grad_rendered_alpha = grad_outputs[1].defined() ? grad_outputs[1].to(torch::kCUDA).contiguous() : torch::Tensor();
            auto grad_radii = grad_outputs[2].defined() ? grad_outputs[2].to(torch::kCUDA).contiguous() : torch::Tensor();
            auto grad_means2d_direct = grad_outputs[3].defined() ? grad_outputs[3].to(torch::kCUDA).contiguous() : torch::Tensor();
            auto grad_depths = grad_outputs[4].defined() ? grad_outputs[4].to(torch::kCUDA).contiguous() : torch::Tensor();

            auto saved = ctx->get_saved_variables();
            auto means3D = saved[0];
            auto quats = saved[1];
            auto scales = saved[2];
            auto sh_coeffs = saved[3];
            auto viewmats = saved[4];
            auto Ks = saved[5];
            auto radii = saved[6];
            auto means2d = saved[7];
            auto depths = saved[8];
            auto conics = saved[9];
            auto compensations = saved[10];
            auto final_opacities = saved[11];
            auto colors = saved[12];
            auto isect_offsets = saved[13];
            auto flatten_ids = saved[14];
            auto rendered_alpha = saved[15];
            auto last_ids = saved[16];

            int image_width = ctx->saved_data["image_width"].to<int>();
            int image_height = ctx->saved_data["image_height"].to<int>();
            int tile_size = ctx->saved_data["tile_size"].to<int>();
            float eps2d = ctx->saved_data["eps2d"].to<float>();
            auto camera_model = static_cast<gsplat::CameraModelType>(
                ctx->saved_data["camera_model"].to<int>());
            bool calc_compensations = ctx->saved_data["calc_compensations"].to<bool>();
            auto bg = ctx->saved_data["bg"].to<torch::Tensor>();
            int sh_degree = ctx->saved_data["sh_degree"].to<int>();

            // Backward through rasterization
            auto raster_grads = gsplat::rasterize_to_pixels_3dgs_bwd(
                means2d,
                conics,
                colors,
                final_opacities,
                bg.defined() ? bg : torch::Tensor(),
                {}, // masks
                image_width,
                image_height,
                tile_size,
                isect_offsets,
                flatten_ids,
                rendered_alpha,
                last_ids,
                grad_rendered_image.defined() ? grad_rendered_image : torch::zeros_like(rendered_alpha),
                grad_rendered_alpha.defined() ? grad_rendered_alpha : torch::zeros_like(rendered_alpha),
                false // absgrad
            );

            auto v_means2d_abs = std::get<0>(raster_grads).contiguous();
            auto v_means2d = std::get<1>(raster_grads).contiguous();
            auto v_conics = std::get<2>(raster_grads).contiguous();
            auto v_colors = std::get<3>(raster_grads).contiguous();
            auto v_opacities = std::get<4>(raster_grads).contiguous();

            // Add direct gradient from means2d if provided
            if (grad_means2d_direct.defined()) {
                v_means2d = v_means2d + grad_means2d_direct;
            }

            // Apply compensation gradients if needed
            torch::Tensor v_compensations = torch::zeros_like(compensations);
            if (calc_compensations) {
                auto base_opacities = final_opacities / compensations;
                v_compensations = v_opacities * base_opacities;
                v_opacities = v_opacities * compensations;
            }

            // Backward through spherical harmonics
            torch::Tensor v_sh_coeffs = torch::zeros_like(sh_coeffs);
            torch::Tensor v_dirs = torch::zeros({viewmats.size(0), means3D.size(0), 3}, means3D.options());

            if (sh_degree > 0 && sh_coeffs.size(1) > 1) {
                auto viewmats_inv = torch::inverse(viewmats);
                auto campos = viewmats_inv.index({Slice(), Slice(None, 3), 3});
                auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1);
                auto masks = (radii > 0).all(-1);

                std::vector<torch::Tensor> v_sh_list;
                for (int c = 0; c < viewmats.size(0); ++c) {
                    auto cam_dirs = dirs[c];
                    auto cam_mask = masks[c];
                    auto cam_v_colors = v_colors[c];

                    auto sh_grads = gsplat::spherical_harmonics_bwd(
                        sh_coeffs.size(1),
                        sh_degree,
                        cam_dirs,
                        sh_coeffs,
                        cam_mask,
                        cam_v_colors,
                        true // compute_v_dirs
                    );

                    v_sh_list.push_back(std::get<1>(sh_grads)); // v_coeffs
                    v_dirs[c] = std::get<0>(sh_grads);          // v_dirs
                }

                // Sum gradients from all cameras
                v_sh_coeffs = torch::stack(v_sh_list, 0).sum(0);
            } else {
                // No SH, gradient goes directly to DC component
                v_sh_coeffs.index_put_({Slice(), 0, Slice()}, v_colors.sum(0));
            }

            // v_dirs contributes to v_means3D
            torch::Tensor v_means3D_from_dirs = v_dirs.sum(0);

            // Backward through projection
            auto proj_grads = gsplat::projection_ewa_3dgs_fused_bwd(
                means3D,
                {}, // covars
                quats,
                scales,
                viewmats,
                Ks,
                image_width,
                image_height,
                eps2d,
                camera_model,
                radii,
                conics,
                compensations,
                v_means2d,
                grad_depths.defined() ? grad_depths : depths.new_zeros(depths.sizes()),
                v_conics,
                v_compensations,
                viewmats.requires_grad());

            auto v_means3D = std::get<0>(proj_grads) + v_means3D_from_dirs;
            auto v_quats = std::get<2>(proj_grads);
            auto v_scales = std::get<3>(proj_grads);
            auto v_viewmats = std::get<4>(proj_grads);

            // Sum opacity gradients from all cameras
            if (v_opacities.dim() == 2) {         // [C, N]
                v_opacities = v_opacities.sum(0); // [N]
            }

            torch::autograd::tensor_list grads;
            grads.push_back(v_means3D);
            grads.push_back(v_quats);
            grads.push_back(v_scales);
            grads.push_back(v_opacities);
            grads.push_back(v_sh_coeffs);
            grads.push_back(v_viewmats);
            grads.push_back(torch::Tensor()); // Ks gradient
            grads.push_back(torch::Tensor()); // bg gradient
            grads.push_back(torch::Tensor()); // image_height
            grads.push_back(torch::Tensor()); // image_width
            grads.push_back(torch::Tensor()); // eps2d
            grads.push_back(torch::Tensor()); // near_plane
            grads.push_back(torch::Tensor()); // far_plane
            grads.push_back(torch::Tensor()); // radius_clip
            grads.push_back(torch::Tensor()); // sh_degree
            grads.push_back(torch::Tensor()); // calc_compensations
            grads.push_back(torch::Tensor()); // camera_model
            return grads;
        }
    };

    GSplatRenderOutput render_gsplat(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier,
        bool packed) {

        // Get camera parameters
        int image_height = static_cast<int>(viewpoint_camera.image_height());
        int image_width = static_cast<int>(viewpoint_camera.image_width());

        // Prepare viewmat and K for single camera (add batch dimension)
        auto viewmat = viewpoint_camera.world_view_transform().t().unsqueeze(0);

        float tanfovx = std::tan(viewpoint_camera.FoVx() * 0.5f);
        float tanfovy = std::tan(viewpoint_camera.FoVy() * 0.5f);
        // Extract K from FoV and image dimensions
        const float focal_length_x = viewpoint_camera.image_width() / (2 * tanfovx);
        const float focal_length_y = viewpoint_camera.image_height() / (2 * tanfovy);

        float cx = image_width / 2.0f;
        float cy = image_height / 2.0f;

        auto K = torch::zeros({1, 3, 3}, viewmat.options());
        K[0][0][0] = focal_length_x;
        K[0][1][1] = focal_length_y;
        ;
        K[0][0][2] = cx;
        K[0][1][2] = cy;
        K[0][2][2] = 1.0f;

        // Get Gaussian parameters
        auto means3D = gaussian_model.get_xyz();
        auto opacities = gaussian_model.get_opacity();
        auto scales = gaussian_model.get_scaling() * scaling_modifier;
        auto rotations = gaussian_model.get_rotation();

        // Get SH coefficients
        auto sh_coeffs = gaussian_model.get_features(); // [N, K, 3]
        int sh_degree = gaussian_model.get_active_sh_degree();

        // Ensure background color is properly shaped
        if (!bg_color.defined() || bg_color.numel() == 0) {
            bg_color = torch::zeros({1, 3}, means3D.options());
        } else {
            bg_color = bg_color.view({1, -1});
        }

        // Call forward function
        auto outputs = _GSplatRasterizeGaussians::apply(
            means3D,
            rotations,
            scales,
            opacities,
            sh_coeffs,
            viewmat,
            K,
            bg_color,
            image_height,
            image_width,
            0.3f,   // eps2d
            0.01f,  // near_plane
            100.0f, // far_plane
            0.0f,   // radius_clip
            sh_degree,
            false, // calc_compensations
            0      // camera_model (0 = PINHOLE)
        );

        GSplatRenderOutput result;
        result.image = outputs[0].squeeze(0).permute({2, 0, 1}); // [C, H, W, 3] -> [3, H, W]
        result.means2d = outputs[3].squeeze(0);                  // [C, N, 2] -> [N, 2]
        result.depths = outputs[4].squeeze(0);                   // [C, N] -> [N]
        result.radii = outputs[2].squeeze(0);                    // [C, N, 2] -> [N, 2]

        // Ensure proper dimensions for visibility mask computation
        if (result.radii.dim() == 1) {
            result.radii = result.radii.unsqueeze(-1).expand({-1, 2});
        }

        // For packed mode, these would be populated
        result.camera_ids = torch::Tensor();
        result.gaussian_ids = torch::Tensor();

        result.means2d.requires_grad_(true);
        result.means2d.retain_grad();
        return result;
    }

} // namespace gs