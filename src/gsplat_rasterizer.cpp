#include "core/gsplat_rasterizer.hpp"
#include "Ops.h"
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

            // Convert camera model int to enum
            auto camera_model = static_cast<gsplat::CameraModelType>(camera_model_int);

            // Get dimensions
            int N = means3D.size(0);
            int C = viewmats.size(0);

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

            auto radii = std::get<0>(proj_results);
            auto means2d = std::get<1>(proj_results);
            auto depths = std::get<2>(proj_results);
            auto conics = std::get<3>(proj_results);
            auto compensations = std::get<4>(proj_results);

            // Apply compensations to opacities if calculated
            torch::Tensor final_opacities = opacities.unsqueeze(0).expand({C, N});
            if (calc_compensations && compensations.defined()) {
                final_opacities = final_opacities * compensations;
            }

            // Step 2: Tile intersection
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

            // Step 3: Encode offsets
            auto isect_offsets = gsplat::intersect_offset(
                isect_ids,
                C,
                tile_width,
                tile_height);

            // Prepare colors - expand to [C, N, channels] if needed
            torch::Tensor render_colors = colors_precomp;
            if (render_colors.dim() == 2) {
                render_colors = render_colors.unsqueeze(0).expand({C, N, -1});
            }

            // Step 4: Rasterize to pixels
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

            auto rendered_image = std::get<0>(raster_results);
            auto rendered_alpha = std::get<1>(raster_results);
            auto last_ids = std::get<2>(raster_results);

            // Save for backward
            ctx->save_for_backward({means3D, quats, scales, colors_precomp, viewmats, Ks,
                                    radii, means2d, depths, conics, compensations, final_opacities,
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

            auto grad_rendered_image = grad_outputs[0];
            auto grad_rendered_alpha = grad_outputs[1];

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
            auto raster_grads = gsplat::rasterize_to_pixels_3dgs_bwd(
                means2d,
                conics,
                colors_precomp.dim() == 3 ? colors_precomp : colors_precomp.unsqueeze(0).expand({viewmats.size(0), -1, -1}),
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
                grad_rendered_image.contiguous(),
                grad_rendered_alpha.contiguous(),
                false // absgrad
            );

            auto v_means2d = std::get<0>(raster_grads);
            auto v_conics = std::get<1>(raster_grads);
            auto v_colors = std::get<2>(raster_grads);
            auto v_opacities = std::get<3>(raster_grads);
            // auto v_background = std::get<4>(raster_grads);

            // Apply compensation gradients if needed
            torch::Tensor v_compensations;
            if (calc_compensations && compensations.defined()) {
                // v_opacities contains gradient w.r.t (opacity * compensation)
                // We need to split this into v_opacity and v_compensation
                auto base_opacities = saved[3].unsqueeze(0).expand({viewmats.size(0), -1});
                v_compensations = v_opacities * base_opacities;
                v_opacities = v_opacities * compensations;
            }

            // Step 2: Backward through projection
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
                v_means2d.contiguous(),
                grad_outputs[4].defined() ? grad_outputs[4].contiguous() : depths.new_zeros(depths.sizes()),
                v_conics.contiguous(),
                v_compensations.defined() ? v_compensations.contiguous() : torch::Tensor(),
                ctx->needs_input_grad(5) // viewmats_requires_grad
            );

            auto v_means3D = std::get<0>(proj_grads);
            // auto v_covars = std::get<1>(proj_grads);  // unused since we use quats/scales
            auto v_quats = std::get<2>(proj_grads);
            auto v_scales = std::get<3>(proj_grads);
            auto v_viewmats = std::get<4>(proj_grads);

            // Sum opacity gradients from all cameras
            if (v_opacities.dim() == 2) {         // [C, N]
                v_opacities = v_opacities.sum(0); // [N]
            }

            // Handle color gradients
            if (colors_precomp.dim() == 2 && v_colors.dim() == 3) {
                v_colors = v_colors.sum(0); // Sum over cameras
            }

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

        // Get camera parameters
        int image_height = static_cast<int>(viewpoint_camera.image_height());
        int image_width = static_cast<int>(viewpoint_camera.image_width());

        // Prepare viewmat and K for single camera (add batch dimension)
        auto viewmat = viewpoint_camera.world_view_transform().unsqueeze(0);

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

        // Get Gaussian parameters
        auto means3D = gaussian_model.get_xyz();
        auto opacities = gaussian_model.get_opacity();
        auto scales = gaussian_model.get_scaling() * scaling_modifier;
        auto rotations = gaussian_model.get_rotation();

        // Get colors (assuming RGB for now, no SH)
        auto colors = gaussian_model.get_features();
        if (colors.size(-1) > 3) {
            // If SH coefficients, just take DC component for now
            colors = colors.index({Slice(), Slice(None, 3)});
        }

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
        result.image = outputs[0].squeeze(0).permute({2, 0, 1}); // [H, W, 3] -> [3, H, W]
        result.means2d = outputs[3].squeeze(0);                  // [N, 2]
        result.depths = outputs[4].squeeze(0);                   // [N]
        result.radii = outputs[2].squeeze(0);                    // [N, 2]

        // For packed mode, these would be populated
        result.camera_ids = torch::Tensor();
        result.gaussian_ids = torch::Tensor();

        return result;
    }

} // namespace gs