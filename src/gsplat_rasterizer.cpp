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
            torch::Tensor colors_precomp,
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

            // Debug input tensors
            INSPECT_TENSOR(means3D);
            INSPECT_TENSOR(quats);
            INSPECT_TENSOR(scales);
            INSPECT_TENSOR(opacities);
            INSPECT_TENSOR(colors_precomp);
            INSPECT_TENSOR(viewmats);
            INSPECT_TENSOR(Ks);
            if (bg.defined()) {
                INSPECT_TENSOR(bg);
            }

            // Validate inputs
            VALIDATE_TENSOR(means3D);
            VALIDATE_TENSOR(quats);
            VALIDATE_TENSOR(scales);
            VALIDATE_TENSOR(opacities);
            VALIDATE_TENSOR(colors_precomp);
            VALIDATE_TENSOR(viewmats);
            VALIDATE_TENSOR(Ks);

            // Convert camera model int to enum
            auto camera_model = static_cast<gsplat::CameraModelType>(camera_model_int);

            // Get dimensions
            int N = means3D.size(0);
            int C = viewmats.size(0);

            std::cout << ts::color::CYAN << "GSplat Forward - N: " << N << ", C: " << C
                      << ", Image: " << image_width << "x" << image_height << ts::color::RESET << std::endl;

            // Ensure all inputs are on CUDA and contiguous
            means3D = means3D.to(torch::kCUDA).contiguous();
            quats = quats.to(torch::kCUDA).contiguous();
            scales = scales.to(torch::kCUDA).contiguous();
            opacities = opacities.to(torch::kCUDA).contiguous();
            viewmats = viewmats.to(torch::kCUDA).contiguous();
            Ks = Ks.to(torch::kCUDA).contiguous();
            if (bg.defined()) {
                bg = bg.to(torch::kCUDA).contiguous();
            }

            // Step 1: Projection using gsplat's fully fused projection
            std::cout << ts::color::YELLOW << "Calling projection..." << ts::color::RESET << std::endl;
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

            // Debug projection outputs
            INSPECT_TENSOR(radii);
            INSPECT_TENSOR(means2d);
            INSPECT_TENSOR(depths);
            INSPECT_TENSOR(conics);

            // Ensure compensations is always a valid tensor
            if (!compensations.defined()) {
                compensations = torch::ones({C, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            } else {
                compensations = compensations.contiguous();
            }
            INSPECT_TENSOR(compensations);

            // Apply compensations to opacities if calculated
            torch::Tensor final_opacities = opacities.unsqueeze(0).expand({C, N}).to(torch::kCUDA).contiguous();
            if (calc_compensations) {
                final_opacities = final_opacities * compensations;
            }
            INSPECT_TENSOR(final_opacities);

            // Step 2: Tile intersection
            int tile_size = 16;
            int tile_width = (image_width + tile_size - 1) / tile_size;
            int tile_height = (image_height + tile_size - 1) / tile_size;

            std::cout << ts::color::YELLOW << "Calling tile intersection..." << ts::color::RESET << std::endl;
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

            INSPECT_TENSOR(tiles_per_gauss);
            INSPECT_TENSOR(isect_ids);
            INSPECT_TENSOR(flatten_ids);

            // Step 3: Encode offsets
            auto isect_offsets = gsplat::intersect_offset(
                isect_ids,
                C,
                tile_width,
                tile_height);
            INSPECT_TENSOR(isect_offsets);

            // Prepare colors - expand to [C, N, channels] if needed
            torch::Tensor render_colors = colors_precomp.to(torch::kCUDA).contiguous();
            if (render_colors.dim() == 2) {
                std::cout << ts::color::BLUE << "Expanding colors from [N, D] to [C, N, D]" << ts::color::RESET << std::endl;
                render_colors = render_colors.unsqueeze(0).expand({C, N, -1}).contiguous();
            }
            INSPECT_TENSOR(render_colors);

            // Step 4: Rasterize to pixels
            std::cout << ts::color::YELLOW << "Calling rasterization..." << ts::color::RESET << std::endl;
            auto raster_results = gsplat::rasterize_to_pixels_3dgs_fwd(
                means2d,
                conics,
                render_colors,
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

            INSPECT_TENSOR(rendered_image);
            INSPECT_TENSOR(rendered_alpha);
            INSPECT_TENSOR(last_ids);

            // Save for backward
            ctx->save_for_backward({means3D, quats, scales, colors_precomp, viewmats, Ks,
                                    radii, means2d, depths, conics,
                                    compensations,
                                    final_opacities,
                                    isect_offsets, flatten_ids, rendered_alpha, last_ids});

            ctx->saved_data["image_width"] = image_width;
            ctx->saved_data["image_height"] = image_height;
            ctx->saved_data["tile_size"] = tile_size;
            ctx->saved_data["eps2d"] = eps2d;
            ctx->saved_data["camera_model"] = camera_model_int;
            ctx->saved_data["calc_compensations"] = calc_compensations;
            ctx->saved_data["bg"] = bg;

            return {rendered_image, rendered_alpha, radii, means2d, depths};
        }

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {

            std::cout << ts::color::CYAN << "GSplat Backward" << ts::color::RESET << std::endl;

            // Debug gradient inputs
            for (size_t i = 0; i < grad_outputs.size(); ++i) {
                if (grad_outputs[i].defined()) {
                    std::cout << "grad_output[" << i << "]: " << grad_outputs[i].sizes() << std::endl;
                }
            }

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
            auto colors_precomp = saved[3];
            auto viewmats = saved[4];
            auto Ks = saved[5];
            auto radii = saved[6];
            auto means2d = saved[7];
            auto depths = saved[8];
            auto conics = saved[9];
            auto compensations = saved[10];
            auto final_opacities = saved[11];
            auto isect_offsets = saved[12];
            auto flatten_ids = saved[13];
            auto rendered_alpha = saved[14];
            auto last_ids = saved[15];

            int image_width = ctx->saved_data["image_width"].to<int>();
            int image_height = ctx->saved_data["image_height"].to<int>();
            int tile_size = ctx->saved_data["tile_size"].to<int>();
            float eps2d = ctx->saved_data["eps2d"].to<float>();
            auto camera_model = static_cast<gsplat::CameraModelType>(
                ctx->saved_data["camera_model"].to<int>());
            bool calc_compensations = ctx->saved_data["calc_compensations"].to<bool>();
            auto bg = ctx->saved_data["bg"].to<torch::Tensor>();

            // Step 1: Backward through rasterization
            std::cout << ts::color::YELLOW << "Backward through rasterization..." << ts::color::RESET << std::endl;

            // Prepare colors for backward - ensure correct dimensions
            torch::Tensor colors_for_bwd = colors_precomp;
            if (colors_precomp.dim() == 2) {
                colors_for_bwd = colors_precomp.unsqueeze(0).expand({viewmats.size(0), -1, -1});
            }
            INSPECT_TENSOR(colors_for_bwd);

            auto raster_grads = gsplat::rasterize_to_pixels_3dgs_bwd(
                means2d,
                conics,
                colors_for_bwd,
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

            INSPECT_TENSOR(v_means2d);
            INSPECT_TENSOR(v_conics);
            INSPECT_TENSOR(v_colors);
            INSPECT_TENSOR(v_opacities);

            // Add direct gradient from means2d if provided
            if (grad_means2d_direct.defined()) {
                v_means2d = v_means2d + grad_means2d_direct;
            }

            // Apply compensation gradients if needed
            torch::Tensor v_compensations = torch::zeros_like(compensations);
            if (calc_compensations) {
                // v_opacities contains gradient w.r.t (opacity * compensation)
                // We need to split this into v_opacity and v_compensation
                auto base_opacities = final_opacities / compensations;
                v_compensations = v_opacities * base_opacities;
                v_opacities = v_opacities * compensations;
            }

            // Step 2: Backward through projection
            std::cout << ts::color::YELLOW << "Backward through projection..." << ts::color::RESET << std::endl;
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

            auto v_means3D = std::get<0>(proj_grads);
            auto v_quats = std::get<2>(proj_grads);
            auto v_scales = std::get<3>(proj_grads);
            auto v_viewmats = std::get<4>(proj_grads);

            INSPECT_TENSOR(v_means3D);
            INSPECT_TENSOR(v_quats);
            INSPECT_TENSOR(v_scales);

            // Sum opacity gradients from all cameras
            if (v_opacities.dim() == 2) { // [C, N]
                std::cout << ts::color::BLUE << "Summing opacity gradients over cameras" << ts::color::RESET << std::endl;
                v_opacities = v_opacities.sum(0); // [N]
            }
            INSPECT_TENSOR(v_opacities);

            // Handle color gradients
            if (colors_precomp.dim() == 2 && v_colors.dim() == 3) {
                std::cout << ts::color::BLUE << "Summing color gradients over cameras" << ts::color::RESET << std::endl;
                v_colors = v_colors.sum(0); // Sum over cameras
            }
            INSPECT_TENSOR(v_colors);

            torch::autograd::tensor_list grads;
            grads.push_back(v_means3D);
            grads.push_back(v_quats);
            grads.push_back(v_scales);
            grads.push_back(v_opacities);
            grads.push_back(v_colors);
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

        std::cout << ts::color::GREEN << "\n=== GSplat Render ===" << ts::color::RESET << std::endl;

        // Get camera parameters
        int image_height = static_cast<int>(viewpoint_camera.image_height());
        int image_width = static_cast<int>(viewpoint_camera.image_width());

        // Prepare viewmat and K for single camera (add batch dimension)
        auto viewmat = viewpoint_camera.world_view_transform().unsqueeze(0);
        INSPECT_TENSOR(viewmat);

        // Extract K from FoV and image dimensions
        float fx = image_width / (2.0f * std::tan(viewpoint_camera.FoVx() / 2.0f));
        float fy = image_height / (2.0f * std::tan(viewpoint_camera.FoVy() / 2.0f));
        float cx = image_width / 2.0f;
        float cy = image_height / 2.0f;

        auto K = torch::zeros({1, 3, 3}, viewmat.options());
        K[0][0][0] = fx;
        K[0][1][1] = fy;
        K[0][0][2] = cx;
        K[0][1][2] = cy;
        K[0][2][2] = 1.0f;
        INSPECT_TENSOR(K);

        // Get Gaussian parameters
        auto means3D = gaussian_model.get_xyz();
        auto opacities = gaussian_model.get_opacity();
        auto scales = gaussian_model.get_scaling() * scaling_modifier;
        auto rotations = gaussian_model.get_rotation();

        std::cout << "Gaussian count: " << means3D.size(0) << std::endl;
        INSPECT_TENSOR(means3D);
        INSPECT_TENSOR(opacities);
        INSPECT_TENSOR(scales);
        INSPECT_TENSOR(rotations);

        // Get colors (assuming RGB for now, no SH)
        auto colors = gaussian_model.get_features();
        INSPECT_TENSOR(colors);

        if (colors.size(-1) > 3) {
            // If SH coefficients, just take DC component for now
            std::cout << ts::color::YELLOW << "Extracting DC component from SH coefficients" << ts::color::RESET << std::endl;
            colors = colors.index({Slice(), Slice(None, 3)});
        }
        INSPECT_TENSOR(colors);

        // Ensure background color is properly shaped
        if (!bg_color.defined() || bg_color.numel() == 0) {
            bg_color = torch::zeros({1, 3}, means3D.options());
        } else {
            bg_color = bg_color.view({1, -1});
        }
        INSPECT_TENSOR(bg_color);

        // Call forward function
        auto outputs = _GSplatRasterizeGaussians::apply(
            means3D,
            rotations,
            scales,
            opacities,
            colors,
            viewmat,
            K,
            bg_color,
            image_height,
            image_width,
            0.3f,   // eps2d
            0.01f,  // near_plane
            100.0f, // far_plane
            0.0f,   // radius_clip
            0,      // sh_degree (0 for no SH)
            false,  // calc_compensations
            0       // camera_model (0 = PINHOLE)
        );

        GSplatRenderOutput result;
        result.image = outputs[0].squeeze(0).permute({2, 0, 1}); // [C, H, W, 3] -> [3, H, W]
        result.means2d = outputs[3].squeeze(0);                  // [C, N, 2] -> [N, 2]
        result.depths = outputs[4].squeeze(0);                   // [C, N] -> [N]
        result.radii = outputs[2].squeeze(0);                    // [C, N, 2] -> [N, 2]

        // Ensure proper dimensions for visibility mask computation
        if (result.radii.dim() == 1) {
            // If radii is 1D, we need to handle it differently
            std::cout << ts::color::YELLOW << "Warning: radii is 1D, reshaping..." << ts::color::RESET << std::endl;
            result.radii = result.radii.unsqueeze(-1).expand({-1, 2});
        }

        INSPECT_TENSOR(result.image);
        INSPECT_TENSOR(result.means2d);
        INSPECT_TENSOR(result.depths);
        INSPECT_TENSOR(result.radii);

        // For packed mode, these would be populated
        result.camera_ids = torch::Tensor();
        result.gaussian_ids = torch::Tensor();

        std::cout << ts::color::GREEN << "=== GSplat Render Complete ===" << ts::color::RESET << std::endl;

        result.means2d.requires_grad_(true);
        result.means2d.retain_grad();
        return result;
    }

} // namespace gs