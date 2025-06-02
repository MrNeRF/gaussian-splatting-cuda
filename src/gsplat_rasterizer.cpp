#include "core/gsplat_rasterizer.hpp"
#include "Ops.h"
#include "core/debug_utils.hpp"
#include <cmath>
#include <torch/torch.h>

namespace gs {

    using namespace torch::indexing;

    // Custom autograd function for the entire rendering pipeline
    class GSplatRenderFunction : public torch::autograd::Function<GSplatRenderFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            // Gaussian parameters
            torch::Tensor means3D,      // [N, 3]
            torch::Tensor quats,        // [N, 4]
            torch::Tensor scales,       // [N, 3]
            torch::Tensor opacities,    // [N]
            torch::Tensor sh_coeffs,    // [N, K, 3]
            // Camera parameters
            torch::Tensor viewmat,      // [1, 4, 4]
            torch::Tensor K,            // [1, 3, 3]
            torch::Tensor bg_color,     // [1, 3]
            // Render settings as tensors
            torch::Tensor settings,     // [9] tensor containing: width, height, sh_degree, etc.
            // Pre-allocated means2d that requires grad
            torch::Tensor means2d_input) {  // [N, 2] - passed in to maintain gradient flow

            // Extract settings
            auto settings_cpu = settings.cpu();
            int width = settings_cpu[0].item<int>();
            int height = settings_cpu[1].item<int>();
            int sh_degree = settings_cpu[2].item<int>();
            float eps2d = settings_cpu[3].item<float>();
            float near_plane = settings_cpu[4].item<float>();
            float far_plane = settings_cpu[5].item<float>();
            float radius_clip = settings_cpu[6].item<float>();
            float scaling_modifier = settings_cpu[7].item<float>();
            int tile_size = settings_cpu[8].item<int>();

            // Ensure all tensors are on CUDA and contiguous
            means3D = means3D.to(torch::kCUDA).contiguous();
            quats = quats.to(torch::kCUDA).contiguous();
            scales = scales.to(torch::kCUDA).contiguous();
            opacities = opacities.to(torch::kCUDA).contiguous();
            sh_coeffs = sh_coeffs.to(torch::kCUDA).contiguous();
            viewmat = viewmat.to(torch::kCUDA).contiguous();
            K = K.to(torch::kCUDA).contiguous();
            bg_color = bg_color.to(torch::kCUDA).contiguous();

            int C = 1;  // Single camera
            int N = means3D.size(0);

            // Apply scaling modifier
            auto scaled_scales = scales * scaling_modifier;

            // Step 1: Projection
            auto proj_results = gsplat::projection_ewa_3dgs_fused_fwd(
                means3D,
                {},  // covars
                quats,
                scaled_scales,
                opacities,
                viewmat,
                K,
                width,
                height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                false,  // calc_compensations
                gsplat::CameraModelType::PINHOLE);

            auto radii = std::get<0>(proj_results).contiguous();        // [C, N, 2]
            auto means2d_proj = std::get<1>(proj_results).contiguous(); // [C, N, 2]
            auto depths = std::get<2>(proj_results).contiguous();       // [C, N]
            auto conics = std::get<3>(proj_results).contiguous();       // [C, N, 3]
            auto compensations = std::get<4>(proj_results);

            // Copy projected means to the input means2d to maintain gradient flow
            means2d_input.copy_(means2d_proj.squeeze(0));

            if (!compensations.defined()) {
                compensations = torch::ones({C, N},
                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            }

            // Step 2: Compute colors from SH
            torch::Tensor colors;
            if (sh_degree > 0 && sh_coeffs.size(1) > 1) {
                auto viewmat_inv = torch::inverse(viewmat);
                auto campos = viewmat_inv.index({Slice(), Slice(None, 3), 3});  // [C, 3]
                auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1);         // [C, N, 3]
                auto masks = (radii > 0).all(-1);                               // [C, N]

                // Compute SH for single camera
                colors = gsplat::spherical_harmonics_fwd(
                    sh_degree, dirs[0], sh_coeffs, masks[0]);
                colors = colors.unsqueeze(0);  // [1, N, 3]
                colors = torch::clamp_min(colors + 0.5f, 0.0f);
            } else {
                colors = sh_coeffs.index({Slice(), 0, Slice()});  // [N, 3]
                colors = colors.unsqueeze(0);                      // [1, N, 3]
                colors = torch::clamp_min(colors + 0.5f, 0.0f);
            }

            // Step 3: Apply opacity with compensations
            auto final_opacities = opacities.unsqueeze(0) * compensations;  // [C, N]

            // Step 4: Tile intersection
            int tile_width = (width + tile_size - 1) / tile_size;
            int tile_height = (height + tile_size - 1) / tile_size;

            // Use means2d_input for tile intersection to maintain gradient connection
            auto means2d_for_isect = means2d_input.unsqueeze(0);  // [1, N, 2]

            auto isect_results = gsplat::intersect_tile(
                means2d_for_isect, radii, depths, {}, {},
                C, tile_size, tile_width, tile_height,
                true, false);

            auto tiles_per_gauss = std::get<0>(isect_results);
            auto isect_ids = std::get<1>(isect_results);
            auto flatten_ids = std::get<2>(isect_results);

            auto isect_offsets = gsplat::intersect_offset(
                isect_ids, C, tile_width, tile_height);
            isect_offsets = isect_offsets.reshape({C, tile_height, tile_width});

            // Step 5: Rasterization - use means2d_input to maintain gradient flow
            auto raster_results = gsplat::rasterize_to_pixels_3dgs_fwd(
                means2d_for_isect, conics, colors, final_opacities,
                bg_color, {},  // masks
                width, height, tile_size,
                isect_offsets, flatten_ids);

            auto rendered_image = std::get<0>(raster_results).contiguous();
            auto rendered_alpha = std::get<1>(raster_results).to(torch::kFloat32).contiguous();
            auto last_ids = std::get<2>(raster_results).contiguous();

            // Save for backward
            ctx->save_for_backward({// Inputs
                                    means3D, quats, scaled_scales, sh_coeffs, viewmat, K,
                                    // Intermediate results
                                    radii, means2d_for_isect, depths, conics, compensations, colors, final_opacities,
                                    // Rasterization
                                    isect_offsets, flatten_ids, rendered_alpha, last_ids,
                                    // Settings and background
                                    settings, bg_color
            });

            // Return rendered image, alpha, and radii (means2d is already in means2d_input)
            return {rendered_image, rendered_alpha, depths, radii};
        }

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {

            auto grad_image = grad_outputs[0].to(torch::kCUDA).contiguous();
            auto grad_alpha = grad_outputs[1].to(torch::kCUDA).contiguous();
            auto grad_depths_extra = grad_outputs[2];
            auto grad_radii_extra = grad_outputs[3];

            auto saved = ctx->get_saved_variables();
            auto means3D = saved[0];
            auto quats = saved[1];
            auto scales = saved[2];
            auto sh_coeffs = saved[3];
            auto viewmat = saved[4];
            auto K = saved[5];
            auto radii = saved[6];
            auto means2d = saved[7];
            auto depths = saved[8];
            auto conics = saved[9];
            auto compensations = saved[10];
            auto colors = saved[11];
            auto opacities = saved[12];
            auto isect_offsets = saved[13];
            auto flatten_ids = saved[14];
            auto rendered_alpha = saved[15];
            auto last_ids = saved[16];
            auto settings = saved[17];
            auto bg_color = saved[18];

            // Extract settings
            auto settings_cpu = settings.cpu();
            int width = settings_cpu[0].item<int>();
            int height = settings_cpu[1].item<int>();
            int sh_degree = settings_cpu[2].item<int>();
            float eps2d = settings_cpu[3].item<float>();
            int tile_size = settings_cpu[8].item<int>();

            // Backward through rasterization
            auto raster_grads = gsplat::rasterize_to_pixels_3dgs_bwd(
                means2d, conics, colors, opacities.unsqueeze(0),
                bg_color, {},  // masks
                width, height, tile_size,
                isect_offsets, flatten_ids,
                rendered_alpha, last_ids,
                grad_image, grad_alpha,
                false);  // absgrad

            auto v_means2d_abs = std::get<0>(raster_grads);
            auto v_means2d = std::get<1>(raster_grads).contiguous();
            auto v_conics = std::get<2>(raster_grads).contiguous();
            auto v_colors = std::get<3>(raster_grads).contiguous();
            auto v_opacities = std::get<4>(raster_grads).contiguous();

            // Backward through SH (if used)
            torch::Tensor v_sh_coeffs = torch::zeros_like(sh_coeffs);
            torch::Tensor v_dirs;

            if (sh_degree > 0 && sh_coeffs.size(1) > 1) {
                auto viewmat_inv = torch::inverse(viewmat);
                auto campos = viewmat_inv.index({Slice(), Slice(None, 3), 3});
                auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1);
                auto masks = (radii > 0).all(-1);

                auto sh_grads = gsplat::spherical_harmonics_bwd(
                    sh_coeffs.size(1), sh_degree,
                    dirs[0], sh_coeffs, masks[0],
                    v_colors[0], true);  // compute_v_dirs

                v_sh_coeffs = std::get<1>(sh_grads);
                v_dirs = std::get<0>(sh_grads).unsqueeze(0);  // [1, N, 3]
            } else {
                v_sh_coeffs.index_put_({Slice(), 0, Slice()}, v_colors[0]);
                v_dirs = torch::zeros({1, means3D.size(0), 3}, means3D.options());
            }

            // v_opacities needs to account for compensation
            v_opacities = v_opacities * compensations;
            auto v_compensations = v_opacities * opacities.unsqueeze(0) / compensations;

            // Add depth gradients if provided
            auto v_depths = depths.new_zeros(depths.sizes());
            if (grad_depths_extra.defined()) {
                v_depths = grad_depths_extra.to(torch::kCUDA);
            }

            // Backward through projection
            auto proj_grads = gsplat::projection_ewa_3dgs_fused_bwd(
                means3D, {}, quats, scales,
                viewmat, K,
                width, height, eps2d,
                gsplat::CameraModelType::PINHOLE,
                radii, conics, compensations,
                v_means2d, v_depths, v_conics, v_compensations,
                viewmat.requires_grad());

            auto v_means3D = std::get<0>(proj_grads);
            auto v_quats = std::get<2>(proj_grads);
            auto v_scales = std::get<3>(proj_grads);
            auto v_viewmat = std::get<4>(proj_grads);

            // Add contribution from view directions
            v_means3D = v_means3D + v_dirs.sum(0);

            // Sum opacity gradients across cameras
            v_opacities = v_opacities.sum(0);

            // Background gradient
            torch::Tensor v_bg_color;
            if (bg_color.requires_grad()) {
                auto one_minus_alpha = 1.0f - rendered_alpha;
                v_bg_color = (grad_image * one_minus_alpha).sum({1, 2});
            }

            // Return gradients for all inputs - IMPORTANT: gradient for means2d_input
            return {
                v_means3D,
                v_quats,
                v_scales,
                v_opacities,
                v_sh_coeffs,
                v_viewmat,
                torch::Tensor(),  // K gradient
                v_bg_color,
                torch::Tensor(),  // settings gradient
                v_means2d.squeeze(0)  // means2d_input gradient
            };
        }
    };

    // Main render function
    GSplatRenderOutput render_gsplat(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier,
        bool packed) {

        // Get camera parameters
        int image_height = static_cast<int>(viewpoint_camera.image_height());
        int image_width = static_cast<int>(viewpoint_camera.image_width());

        // Prepare viewmat and K
        auto viewmat = viewpoint_camera.world_view_transform().t().unsqueeze(0);

        float tanfovx = std::tan(viewpoint_camera.FoVx() * 0.5f);
        float tanfovy = std::tan(viewpoint_camera.FoVy() * 0.5f);
        const float focal_length_x = viewpoint_camera.image_width() / (2 * tanfovx);
        const float focal_length_y = viewpoint_camera.image_height() / (2 * tanfovy);

        float cx = image_width / 2.0f;
        float cy = image_height / 2.0f;

        auto K = torch::zeros({1, 3, 3}, viewmat.options());
        K[0][0][0] = focal_length_x;
        K[0][1][1] = focal_length_y;
        K[0][0][2] = cx;
        K[0][1][2] = cy;
        K[0][2][2] = 1.0f;

        // Get Gaussian parameters
        auto means3D = gaussian_model.get_xyz();
        auto opacities = gaussian_model.get_opacity().squeeze(-1);  // Remove last dim if present
        auto scales = gaussian_model.get_scaling();
        auto rotations = gaussian_model.get_rotation();
        auto sh_coeffs = gaussian_model.get_features();
        int sh_degree = gaussian_model.get_active_sh_degree();

        // Ensure background color is properly shaped
        if (!bg_color.defined() || bg_color.numel() == 0) {
            bg_color = torch::zeros({1, 3}, means3D.options());
        } else {
            bg_color = bg_color.view({1, -1}).to(torch::kCUDA);
        }

        // Create settings tensor
        auto settings = torch::tensor({
                                          (float)image_width,
                                          (float)image_height,
                                          (float)sh_degree,
                                          0.3f,      // eps2d
                                          0.01f,     // near_plane
                                          100.0f,    // far_plane
                                          0.0f,      // radius_clip
                                          scaling_modifier,
                                          16.0f      // tile_size
                                      }, torch::TensorOptions().dtype(torch::kFloat32));

        // CRITICAL: Create means2d tensor that will track gradients
        int N = means3D.size(0);
        auto means2d_with_grad = torch::zeros({N, 2},
                                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));

        // CRITICAL: Call retain_grad immediately
        means2d_with_grad.retain_grad();

        // Call the unified autograd function
        auto outputs = GSplatRenderFunction::apply(
            means3D, rotations, scales, opacities, sh_coeffs,
            viewmat, K, bg_color, settings, means2d_with_grad);

        auto rendered_image = outputs[0];
        auto rendered_alpha = outputs[1];
        auto depths = outputs[2];
        auto radii = outputs[3];

        // Prepare output
        GSplatRenderOutput result;
        result.image = rendered_image.squeeze(0).permute({2, 0, 1});  // [C, H, W, 3] -> [3, H, W]
        result.means2d = means2d_with_grad;  // Use the tensor with retained gradients
        result.depths = depths.squeeze(0);   // [C, N] -> [N]
        result.radii = radii.squeeze(0);     // [C, N, 2] -> [N, 2]

        // Empty for single camera rendering
        result.camera_ids = torch::Tensor();
        result.gaussian_ids = torch::Tensor();

        return result;
    }

} // namespace gs