#include "Ops.h"
#include "core/rasterizer.hpp"
#include <torch/torch.h>

namespace gs {

    using namespace torch::indexing;

    // Custom autograd function for the entire rendering pipeline
    class GSplatRenderFunction : public torch::autograd::Function<GSplatRenderFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            // Gaussian parameters
            torch::Tensor means3D,   // [N, 3]
            torch::Tensor quats,     // [N, 4]
            torch::Tensor scales,    // [N, 3]
            torch::Tensor opacities, // [N]
            torch::Tensor sh_coeffs, // [N, K, 3]
            // Camera parameters
            torch::Tensor viewmat,  // [1, 4, 4]
            torch::Tensor K,        // [1, 3, 3]
            torch::Tensor bg_color, // [1, 3]
            // Render settings as tensors
            const torch::Tensor& settings, // [9] tensor containing: width, height, sh_degree, etc.
            // Pre-allocated means2d that requires grad
            const torch::Tensor& means2d_input) { // [N, 2] - passed in to maintain gradient flow

            // Input validation
            const int N = static_cast<int>(means3D.size(0));
            TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3,
                        "means3D must be [N, 3], got ", means3D.sizes());
            TORCH_CHECK(quats.dim() == 2 && quats.size(0) == N && quats.size(1) == 4,
                        "quats must be [N, 4], got ", quats.sizes());
            TORCH_CHECK(scales.dim() == 2 && scales.size(0) == N && scales.size(1) == 3,
                        "scales must be [N, 3], got ", scales.sizes());
            TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == N,
                        "opacities must be [N], got ", opacities.sizes());
            TORCH_CHECK(sh_coeffs.dim() == 3 && sh_coeffs.size(0) == N && sh_coeffs.size(2) == 3,
                        "sh_coeffs must be [N, K, 3], got ", sh_coeffs.sizes());
            TORCH_CHECK(viewmat.dim() == 3 && viewmat.size(0) == 1 && viewmat.size(1) == 4 && viewmat.size(2) == 4,
                        "viewmat must be [1, 4, 4], got ", viewmat.sizes());
            TORCH_CHECK(K.dim() == 3 && K.size(0) == 1 && K.size(1) == 3 && K.size(2) == 3,
                        "K must be [1, 3, 3], got ", K.sizes());
            TORCH_CHECK(bg_color.dim() == 2 && bg_color.size(0) == 1 && bg_color.size(1) == 3,
                        "bg_color must be [1, 3], got ", bg_color.sizes());
            TORCH_CHECK(settings.dim() == 1 && settings.size(0) == 9,
                        "settings must be [9], got ", settings.sizes());
            TORCH_CHECK(means2d_input.dim() == 2 && means2d_input.size(0) == N && means2d_input.size(1) == 2,
                        "means2d_input must be [N, 2], got ", means2d_input.sizes());

            // Check device and dtype
            TORCH_CHECK(means3D.is_cuda(), "means3D must be on CUDA");
            TORCH_CHECK(means2d_input.requires_grad(), "means2d_input must require gradients");

            // Extract settings
            const auto settings_cpu = settings.cpu();
            const auto width = settings_cpu[0].item<int>();
            const auto height = settings_cpu[1].item<int>();
            const auto sh_degree = settings_cpu[2].item<int>();
            const auto eps2d = settings_cpu[3].item<float>();
            const auto near_plane = settings_cpu[4].item<float>();
            const auto far_plane = settings_cpu[5].item<float>();
            const auto radius_clip = settings_cpu[6].item<float>();
            const auto scaling_modifier = settings_cpu[7].item<float>();
            const auto tile_size = settings_cpu[8].item<int>();

            // Check SH degree validity
            const int expected_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
            TORCH_CHECK(sh_coeffs.size(1) >= expected_sh_coeffs,
                        "sh_coeffs must have at least ", expected_sh_coeffs, " coefficients for sh_degree ",
                        sh_degree, ", got ", sh_coeffs.size(1));

            // Ensure all tensors are on CUDA and contiguous
            means3D = means3D.to(torch::kCUDA).contiguous();
            quats = quats.to(torch::kCUDA).contiguous();
            scales = scales.to(torch::kCUDA).contiguous();
            opacities = opacities.to(torch::kCUDA).contiguous();
            sh_coeffs = sh_coeffs.to(torch::kCUDA).contiguous();
            viewmat = viewmat.to(torch::kCUDA).contiguous();
            K = K.to(torch::kCUDA).contiguous();
            bg_color = bg_color.to(torch::kCUDA).contiguous();

            const int C = 1; // Single camera

            // Apply scaling modifier
            auto scaled_scales = scales * scaling_modifier;
            TORCH_CHECK(scaled_scales.sizes() == scales.sizes(),
                        "scaled_scales shape mismatch: ", scaled_scales.sizes(), " vs ", scales.sizes());

            // Step 1: Projection
            auto proj_results = gsplat::projection_ewa_3dgs_fused_fwd(
                means3D,
                {}, // covars
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
                false, // calc_compensations
                gsplat::CameraModelType::PINHOLE);

            auto radii = std::get<0>(proj_results).contiguous();        // [C, N, 2]
            auto means2d_proj = std::get<1>(proj_results).contiguous(); // [C, N, 2]
            auto depths = std::get<2>(proj_results).contiguous();       // [C, N]
            auto conics = std::get<3>(proj_results).contiguous();       // [C, N, 3]
            auto compensations = std::get<4>(proj_results);

            // Validate projection outputs
            TORCH_CHECK(radii.dim() == 3 && radii.size(0) == C && radii.size(1) == N && radii.size(2) == 2,
                        "radii must be [C, N, 2], got ", radii.sizes());
            TORCH_CHECK(means2d_proj.dim() == 3 && means2d_proj.size(0) == C && means2d_proj.size(1) == N && means2d_proj.size(2) == 2,
                        "means2d_proj must be [C, N, 2], got ", means2d_proj.sizes());
            TORCH_CHECK(depths.dim() == 2 && depths.size(0) == C && depths.size(1) == N,
                        "depths must be [C, N], got ", depths.sizes());
            TORCH_CHECK(conics.dim() == 3 && conics.size(0) == C && conics.size(1) == N && conics.size(2) == 3,
                        "conics must be [C, N, 3], got ", conics.sizes());

            // Copy projected means to the input means2d to maintain gradient flow
            means2d_input.copy_(means2d_proj.squeeze(0));

            if (!compensations.defined()) {
                compensations = torch::ones({C, N},
                                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            }
            TORCH_CHECK(compensations.dim() == 2 && compensations.size(0) == C && compensations.size(1) == N,
                        "compensations must be [C, N], got ", compensations.sizes());

            // Step 2: Compute colors from SH
            torch::Tensor colors;
            torch::Tensor sh_coeffs_used; // Track which coefficients we actually used
            int num_sh_coeffs = 1;        // Default to just DC

            if (sh_degree > 0 && sh_coeffs.size(1) > 1) {
                auto viewmat_inv = torch::inverse(viewmat);
                TORCH_CHECK(viewmat_inv.sizes() == viewmat.sizes(),
                            "viewmat_inv shape mismatch");

                auto campos = viewmat_inv.index({Slice(), Slice(None, 3), 3}); // [C, 3]
                TORCH_CHECK(campos.dim() == 2 && campos.size(0) == C && campos.size(1) == 3,
                            "campos must be [C, 3], got ", campos.sizes());

                auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1); // [C, N, 3]
                TORCH_CHECK(dirs.dim() == 3 && dirs.size(0) == C && dirs.size(1) == N && dirs.size(2) == 3,
                            "dirs must be [C, N, 3], got ", dirs.sizes());

                auto masks = (radii > 0).all(-1); // [C, N]
                TORCH_CHECK(masks.dim() == 2 && masks.size(0) == C && masks.size(1) == N,
                            "masks must be [C, N], got ", masks.sizes());

                // CRITICAL: Only use the coefficients up to (sh_degree+1)^2
                num_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
                sh_coeffs_used = sh_coeffs.index({Slice(), Slice(None, num_sh_coeffs), Slice()}).contiguous();
                TORCH_CHECK(sh_coeffs_used.dim() == 3 && sh_coeffs_used.size(0) == N &&
                                sh_coeffs_used.size(1) == num_sh_coeffs && sh_coeffs_used.size(2) == 3,
                            "sh_coeffs_used must be [N, ", num_sh_coeffs, ", 3], got ", sh_coeffs_used.sizes());

                // Compute SH for single camera with only active coefficients
                colors = gsplat::spherical_harmonics_fwd(
                    sh_degree, dirs[0], sh_coeffs_used, masks[0]);
                TORCH_CHECK(colors.dim() == 2 && colors.size(0) == N && colors.size(1) == 3,
                            "colors from SH must be [N, 3], got ", colors.sizes());

                colors = colors.unsqueeze(0); // [1, N, 3]
                colors = torch::clamp_min(colors + 0.5f, 0.0f);
            } else {
                // Use only DC component
                sh_coeffs_used = sh_coeffs.index({Slice(), Slice(None, 1), Slice()}).contiguous();
                TORCH_CHECK(sh_coeffs_used.dim() == 3 && sh_coeffs_used.size(0) == N &&
                                sh_coeffs_used.size(1) == 1 && sh_coeffs_used.size(2) == 3,
                            "sh_coeffs_used (DC only) must be [N, 1, 3], got ", sh_coeffs_used.sizes());

                colors = sh_coeffs_used.index({Slice(), 0, Slice()}); // [N, 3]
                TORCH_CHECK(colors.dim() == 2 && colors.size(0) == N && colors.size(1) == 3,
                            "colors (DC only) must be [N, 3], got ", colors.sizes());

                colors = colors.unsqueeze(0); // [1, N, 3]
                colors = torch::clamp_min(colors + 0.5f, 0.0f);
            }

            TORCH_CHECK(colors.dim() == 3 && colors.size(0) == C && colors.size(1) == N && colors.size(2) == 3,
                        "colors must be [C, N, 3], got ", colors.sizes());

            // Step 3: Apply opacity with compensations
            auto final_opacities = opacities.unsqueeze(0) * compensations; // [C, N]
            TORCH_CHECK(final_opacities.dim() == 2 && final_opacities.size(0) == C && final_opacities.size(1) == N,
                        "final_opacities must be [C, N], got ", final_opacities.sizes());

            // Step 4: Tile intersection
            const int tile_width = (width + tile_size - 1) / tile_size;
            const int tile_height = (height + tile_size - 1) / tile_size;

            // Use means2d_input for tile intersection to maintain gradient connection
            auto means2d_for_isect = means2d_input.unsqueeze(0); // [1, N, 2]
            TORCH_CHECK(means2d_for_isect.dim() == 3 && means2d_for_isect.size(0) == 1 &&
                            means2d_for_isect.size(1) == N && means2d_for_isect.size(2) == 2,
                        "means2d_for_isect must be [1, N, 2], got ", means2d_for_isect.sizes());

            const auto isect_results = gsplat::intersect_tile(
                means2d_for_isect, radii, depths, {}, {},
                C, tile_size, tile_width, tile_height,
                true, false);

            const auto tiles_per_gauss = std::get<0>(isect_results);
            const auto isect_ids = std::get<1>(isect_results);
            const auto flatten_ids = std::get<2>(isect_results);

            // Validate intersection results
            TORCH_CHECK(tiles_per_gauss.dim() == 2 && tiles_per_gauss.size(0) == C && tiles_per_gauss.size(1) == N,
                        "tiles_per_gauss must be [C, N], got ", tiles_per_gauss.sizes());
            TORCH_CHECK(isect_ids.dim() == 1, "isect_ids must be 1D, got ", isect_ids.dim(), "D");
            TORCH_CHECK(flatten_ids.dim() == 1 && flatten_ids.size(0) == isect_ids.size(0),
                        "flatten_ids must match isect_ids size, got ", flatten_ids.size(0), " vs ", isect_ids.size(0));

            auto isect_offsets = gsplat::intersect_offset(
                isect_ids, C, tile_width, tile_height);
            TORCH_CHECK(isect_offsets.dim() == 3 && isect_offsets.size(0) == C,
                        "isect_offsets must be [C, H, W], got ", isect_offsets.sizes());

            isect_offsets = isect_offsets.reshape({C, tile_height, tile_width});
            TORCH_CHECK(isect_offsets.size(1) == tile_height && isect_offsets.size(2) == tile_width,
                        "isect_offsets reshape failed: ", isect_offsets.sizes());

            // Step 5: Rasterization - use means2d_input to maintain gradient flow
            auto raster_results = gsplat::rasterize_to_pixels_3dgs_fwd(
                means2d_for_isect, conics, colors, final_opacities,
                bg_color, {}, // masks
                width, height, tile_size,
                isect_offsets, flatten_ids);

            auto rendered_image = std::get<0>(raster_results).contiguous();
            auto rendered_alpha = std::get<1>(raster_results).to(torch::kFloat32).contiguous();
            auto last_ids = std::get<2>(raster_results).contiguous();

            // Validate rasterization outputs
            TORCH_CHECK(rendered_image.dim() == 4 && rendered_image.size(0) == C &&
                            rendered_image.size(1) == height && rendered_image.size(2) == width && rendered_image.size(3) == 3,
                        "rendered_image must be [C, H, W, 3], got ", rendered_image.sizes());
            TORCH_CHECK(rendered_alpha.dim() == 4 && rendered_alpha.size(0) == C &&
                            rendered_alpha.size(1) == height && rendered_alpha.size(2) == width && rendered_alpha.size(3) == 1,
                        "rendered_alpha must be [C, H, W, 1], got ", rendered_alpha.sizes());
            TORCH_CHECK(last_ids.dim() == 3 && last_ids.size(0) == C &&
                            last_ids.size(1) == height && last_ids.size(2) == width,
                        "last_ids must be [C, H, W], got ", last_ids.sizes());

            // Save for backward - save both the full sh_coeffs and which ones we used
            ctx->save_for_backward({// Inputs
                                    means3D, quats, scaled_scales, sh_coeffs, viewmat, K,
                                    // Intermediate results
                                    radii, means2d_for_isect, depths, conics, compensations, colors, final_opacities,
                                    // Rasterization
                                    isect_offsets, flatten_ids, rendered_alpha, last_ids,
                                    // Settings and background
                                    settings, bg_color,
                                    // SH tracking
                                    sh_coeffs_used});

            // Store sh_degree and num_coeffs in context for backward
            ctx->saved_data["sh_degree"] = sh_degree;
            ctx->saved_data["num_sh_coeffs"] = num_sh_coeffs;

            // Return rendered image, alpha, and radii (means2d is already in means2d_input)
            return {rendered_image, rendered_alpha, depths, radii};
        }

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {

            auto grad_image = grad_outputs[0].to(torch::kCUDA).contiguous();
            auto grad_alpha = grad_outputs[1].to(torch::kCUDA).contiguous();
            const auto& grad_depths_extra = grad_outputs[2];

            // Validate gradient inputs
            TORCH_CHECK(grad_image.defined(), "grad_image must be defined");
            TORCH_CHECK(grad_alpha.defined(), "grad_alpha must be defined");

            auto saved = ctx->get_saved_variables();
            const auto& means3D = saved[0];
            auto quats = saved[1];
            auto scales = saved[2];
            const auto& sh_coeffs = saved[3];
            const auto& viewmat = saved[4];
            const auto& K = saved[5];
            const auto& radii = saved[6];
            const auto& means2d = saved[7];
            const auto& depths = saved[8];
            const auto& conics = saved[9];
            auto compensations = saved[10];
            const auto& colors = saved[11];
            const auto& opacities = saved[12];
            const auto& isect_offsets = saved[13];
            const auto& flatten_ids = saved[14];
            const auto& rendered_alpha = saved[15];
            const auto& last_ids = saved[16];
            const auto& settings = saved[17];
            auto bg_color = saved[18];
            const auto& sh_coeffs_used = saved[19];

            // Get sh_degree and num_coeffs from context
            const int sh_degree = ctx->saved_data["sh_degree"].to<int>();
            const int num_sh_coeffs = ctx->saved_data["num_sh_coeffs"].to<int>();

            // Extract settings
            auto settings_cpu = settings.cpu();
            const auto width = settings_cpu[0].item<int>();
            const auto height = settings_cpu[1].item<int>();
            const auto eps2d = settings_cpu[3].item<float>();
            const auto tile_size = settings_cpu[8].item<int>();

            const int C = static_cast<int>(viewmat.size(0));
            const int N = static_cast<int>(means3D.size(0));

            // Validate gradient shapes
            TORCH_CHECK(grad_image.dim() == 4 && grad_image.size(0) == C &&
                            grad_image.size(1) == height && grad_image.size(2) == width && grad_image.size(3) == 3,
                        "grad_image must be [C, H, W, 3], got ", grad_image.sizes());
            TORCH_CHECK(grad_alpha.dim() == 4 && grad_alpha.size(0) == C &&
                            grad_alpha.size(1) == height && grad_alpha.size(2) == width && grad_alpha.size(3) == 1,
                        "grad_alpha must be [C, H, W, 1], got ", grad_alpha.sizes());

            // Backward through rasterization
            auto raster_grads = gsplat::rasterize_to_pixels_3dgs_bwd(
                means2d, conics, colors, opacities,
                bg_color, {}, // masks
                width, height, tile_size,
                isect_offsets, flatten_ids,
                rendered_alpha, last_ids,
                grad_image, grad_alpha,
                false); // absgrad

            auto v_means2d_abs = std::get<0>(raster_grads);
            auto v_means2d = std::get<1>(raster_grads).contiguous();
            auto v_conics = std::get<2>(raster_grads).contiguous();
            auto v_colors = std::get<3>(raster_grads).contiguous();
            auto v_opacities = std::get<4>(raster_grads).contiguous();

            // Validate rasterization backward outputs
            TORCH_CHECK(v_means2d.dim() == 3 && v_means2d.size(0) == C && v_means2d.size(1) == N && v_means2d.size(2) == 2,
                        "v_means2d must be [C, N, 2], got ", v_means2d.sizes());
            TORCH_CHECK(v_conics.dim() == 3 && v_conics.size(0) == C && v_conics.size(1) == N && v_conics.size(2) == 3,
                        "v_conics must be [C, N, 3], got ", v_conics.sizes());
            TORCH_CHECK(v_colors.dim() == 3 && v_colors.size(0) == C && v_colors.size(1) == N && v_colors.size(2) == 3,
                        "v_colors must be [C, N, 3], got ", v_colors.sizes());
            TORCH_CHECK(v_opacities.dim() == 2 && v_opacities.size(0) == C && v_opacities.size(1) == N,
                        "v_opacities must be [C, N], got ", v_opacities.sizes());

            // Backward through SH (if used)
            torch::Tensor v_sh_coeffs = torch::zeros_like(sh_coeffs);
            torch::Tensor v_dirs;

            if (sh_degree > 0 && sh_coeffs.size(1) > 1) {
                auto viewmat_inv = torch::inverse(viewmat);
                auto campos = viewmat_inv.index({Slice(), Slice(None, 3), 3});
                auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1);
                auto masks = (radii > 0).all(-1);

                TORCH_CHECK(dirs.dim() == 3 && dirs.size(0) == C && dirs.size(1) == N && dirs.size(2) == 3,
                            "dirs for SH backward must be [C, N, 3], got ", dirs.sizes());
                TORCH_CHECK(masks.dim() == 2 && masks.size(0) == C && masks.size(1) == N,
                            "masks for SH backward must be [C, N], got ", masks.sizes());

                auto sh_grads = gsplat::spherical_harmonics_bwd(
                    num_sh_coeffs, sh_degree,
                    dirs[0], sh_coeffs_used, masks[0],
                    v_colors[0], true); // compute_v_dirs

                // The C++ function returns (v_coeffs, v_dirs) in that order
                auto v_sh_coeffs_active = std::get<0>(sh_grads); // This is v_coeffs [N, K, 3]
                auto v_dirs_from_sh = std::get<1>(sh_grads);     // This is v_dirs [N, 3]

                // Now validate the shapes
                TORCH_CHECK(v_dirs_from_sh.dim() == 2 && v_dirs_from_sh.size(0) == N && v_dirs_from_sh.size(1) == 3,
                            "v_dirs_from_sh must be [N, 3], got ", v_dirs_from_sh.sizes());

                // The SH coefficient gradient should match the expected shape
                TORCH_CHECK(v_sh_coeffs_active.dim() == 3 && v_sh_coeffs_active.size(0) == N &&
                                v_sh_coeffs_active.size(1) == num_sh_coeffs && v_sh_coeffs_active.size(2) == 3,
                            "v_sh_coeffs_active must be [N, K, 3], got ", v_sh_coeffs_active.sizes());

                if (v_sh_coeffs_active.sizes() != sh_coeffs_used.sizes()) {
                    std::cerr << "WARNING: SH gradient shape mismatch!" << std::endl;
                    std::cerr << "Expected: " << sh_coeffs_used.sizes() << std::endl;
                    std::cerr << "Got: " << v_sh_coeffs_active.sizes() << std::endl;
                    // This shouldn't happen anymore
                    TORCH_CHECK(false, "SH gradient shape mismatch should not occur");
                }

                v_sh_coeffs.index_put_({Slice(), Slice(None, num_sh_coeffs), Slice()}, v_sh_coeffs_active);

                v_dirs = v_dirs_from_sh.unsqueeze(0); // [1, N, 3]
                TORCH_CHECK(v_dirs.dim() == 3 && v_dirs.size(0) == 1 && v_dirs.size(1) == N && v_dirs.size(2) == 3,
                            "v_dirs must be [1, N, 3], got ", v_dirs.sizes());
            } else {
                // Only DC component
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
                TORCH_CHECK(v_depths.sizes() == depths.sizes(),
                            "v_depths shape mismatch: ", v_depths.sizes(), " vs ", depths.sizes());
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

            // Validate projection backward outputs
            if (v_means3D.defined()) {
                TORCH_CHECK(v_means3D.sizes() == means3D.sizes(),
                            "v_means3D shape mismatch: ", v_means3D.sizes(), " vs ", means3D.sizes());
            }
            if (v_quats.defined()) {
                TORCH_CHECK(v_quats.sizes() == quats.sizes(),
                            "v_quats shape mismatch: ", v_quats.sizes(), " vs ", quats.sizes());
            }
            if (v_scales.defined()) {
                TORCH_CHECK(v_scales.sizes() == scales.sizes(),
                            "v_scales shape mismatch: ", v_scales.sizes(), " vs ", scales.sizes());
            }

            // Add contribution from view directions
            v_means3D = v_means3D + v_dirs.sum(0);

            // Sum opacity gradients across cameras
            v_opacities = v_opacities.sum(0);
            TORCH_CHECK(v_opacities.dim() == 1 && v_opacities.size(0) == N,
                        "v_opacities after sum must be [N], got ", v_opacities.sizes());

            // Background gradient
            torch::Tensor v_bg_color;
            if (bg_color.requires_grad()) {
                auto one_minus_alpha = 1.0f - rendered_alpha;
                v_bg_color = (grad_image * one_minus_alpha).sum({1, 2});
                TORCH_CHECK(v_bg_color.dim() == 2 && v_bg_color.size(0) == C && v_bg_color.size(1) == 3,
                            "v_bg_color must be [C, 3], got ", v_bg_color.sizes());
            }

            // Final means2d gradient
            auto v_means2d_final = v_means2d.squeeze(0);
            TORCH_CHECK(v_means2d_final.dim() == 2 && v_means2d_final.size(0) == N && v_means2d_final.size(1) == 2,
                        "v_means2d_final must be [N, 2], got ", v_means2d_final.sizes());

            // Return gradients for all inputs - IMPORTANT: gradient for means2d_input
            return {
                v_means3D,
                v_quats,
                v_scales,
                v_opacities,
                v_sh_coeffs,
                v_viewmat,
                torch::Tensor(), // K gradient
                v_bg_color,
                torch::Tensor(), // settings gradient
                v_means2d_final  // means2d_input gradient
            };
        }
    };

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
        auto viewmat = viewpoint_camera.world_view_transform();
        TORCH_CHECK(viewmat.dim() == 3 && viewmat.size(0) == 1 && viewmat.size(1) == 4 && viewmat.size(2) == 4,
                    "viewmat must be [1, 4, 4] after transpose and unsqueeze, got ", viewmat.sizes());
        const auto K = viewpoint_camera.K();

        // Get Gaussian parameters
        auto means3D = gaussian_model.get_xyz();
        auto opacities = gaussian_model.get_opacity();
        if (opacities.dim() == 2 && opacities.size(1) == 1) {
            opacities = opacities.squeeze(-1); // Remove last dim if present
        }
        const auto scales = gaussian_model.get_scaling();
        const auto rotations = gaussian_model.get_rotation();
        const auto sh_coeffs = gaussian_model.get_features();
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

        // Ensure background color is properly shaped
        if (!bg_color.defined() || bg_color.numel() == 0) {
            bg_color = torch::zeros({1, 3}, means3D.options());
        } else {
            bg_color = bg_color.view({1, -1}).to(torch::kCUDA);
            TORCH_CHECK(bg_color.size(0) == 1 && bg_color.size(1) == 3,
                        "bg_color must be reshapeable to [1, 3], got ", bg_color.sizes());
        }

        // Create settings tensor
        auto settings = torch::tensor({
                                          (float)image_width,
                                          (float)image_height,
                                          (float)sh_degree,
                                          0.3f,   // eps2d
                                          0.01f,  // near_plane
                                          100.0f, // far_plane
                                          0.0f,   // radius_clip
                                          scaling_modifier,
                                          16.0f // tile_size
                                      },
                                      torch::TensorOptions().dtype(torch::kFloat32));

        // CRITICAL: Create means2d tensor that will track gradients
        auto means2d_with_grad = torch::zeros({N, 2},
                                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));

        // CRITICAL: Call retain_grad immediately
        means2d_with_grad.retain_grad();

        // Call the unified autograd function
        auto outputs = GSplatRenderFunction::apply(
            means3D, rotations, scales, opacities, sh_coeffs,
            viewmat, K, bg_color, settings, means2d_with_grad);

        const auto& rendered_image = outputs[0];
        const auto& rendered_alpha = outputs[1];
        const auto& depths = outputs[2];
        const auto& radii = outputs[3];

        // Validate outputs
        TORCH_CHECK(rendered_image.dim() == 4 && rendered_image.size(0) == 1 &&
                        rendered_image.size(1) == image_height && rendered_image.size(2) == image_width &&
                        rendered_image.size(3) == 3,
                    "rendered_image must be [1, H, W, 3], got ", rendered_image.sizes());
        TORCH_CHECK(rendered_alpha.dim() == 4 && rendered_alpha.size(0) == 1 &&
                        rendered_alpha.size(1) == image_height && rendered_alpha.size(2) == image_width &&
                        rendered_alpha.size(3) == 1,
                    "rendered_alpha must be [1, H, W, 1], got ", rendered_alpha.sizes());
        TORCH_CHECK(depths.dim() == 2 && depths.size(0) == 1 && depths.size(1) == N,
                    "depths must be [1, N], got ", depths.sizes());
        TORCH_CHECK(radii.dim() == 3 && radii.size(0) == 1 && radii.size(1) == N && radii.size(2) == 2,
                    "radii must be [1, N, 2], got ", radii.sizes());

        // Prepare output
        RenderOutput result;
        result.image = rendered_image.squeeze(0).permute({2, 0, 1}); // [C, H, W, 3] -> [3, H, W]
        TORCH_CHECK(result.image.dim() == 3 && result.image.size(0) == 3 &&
                        result.image.size(1) == image_height && result.image.size(2) == image_width,
                    "result.image must be [3, H, W], got ", result.image.sizes());

        result.means2d = means2d_with_grad;                   // Use the tensor with retained gradients
        result.depths = depths.squeeze(0);                    // [C, N] -> [N]
        result.radii = std::get<0>(radii.squeeze(0).max(-1)); // [C, N, 2] -> [N]
        result.visibility = (result.radii > 0);               // any(-1) reduces [N, 2] to [N]

        result.width = image_width;
        result.height = image_height;
        return result;
    }

} // namespace gs