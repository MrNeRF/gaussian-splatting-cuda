#include "rasterization/rasterizer_autograd.hpp"
#include "Projection.h"

namespace gs::training {
    using namespace torch::indexing;

    // SphericalHarmonicsFunction implementation
    torch::autograd::tensor_list SphericalHarmonicsFunction::forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor sh_degree_tensor, // [1] containing sh_degree
        torch::Tensor dirs,             // [..., 3]
        torch::Tensor coeffs,           // [..., K, 3]
        torch::Tensor masks) {
        // [...] optional

        const int sh_degree = sh_degree_tensor.item<int>();
        const int num_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);

        // Input validation
        TORCH_CHECK(dirs.size(-1) == 3,
                    "dirs last dimension must be 3, got ", dirs.size(-1));
        TORCH_CHECK(coeffs.size(-1) == 3,
                    "coeffs last dimension must be 3, got ", coeffs.size(-1));
        TORCH_CHECK(coeffs.size(-2) >= num_sh_coeffs,
                    "coeffs K dimension must be at least ", num_sh_coeffs, ", got ", coeffs.size(-2));

        // Get batch dimensions
        auto batch_dims = dirs.sizes().slice(0, dirs.dim() - 1);

        TORCH_CHECK(dirs.sizes().slice(0, dirs.dim() - 1) == coeffs.sizes().slice(0, coeffs.dim() - 2),
                    "dirs and coeffs batch dimensions must match");

        if (masks.defined()) {
            TORCH_CHECK(masks.sizes() == batch_dims,
                        "masks must match dirs batch dims, got ", masks.sizes());
        }

        // Device checks
        TORCH_CHECK(dirs.is_cuda(), "dirs must be on CUDA");
        TORCH_CHECK(coeffs.is_cuda(), "coeffs must be on CUDA");
        TORCH_CHECK(sh_degree_tensor.is_cuda(), "sh_degree_tensor must be on CUDA");
        if (masks.defined()) {
            TORCH_CHECK(masks.is_cuda(), "masks must be on CUDA");
        }

        // Ensure tensors are contiguous
        dirs = dirs.contiguous();
        coeffs = coeffs.contiguous();
        if (masks.defined()) {
            masks = masks.contiguous();
        } else {
            // Create default masks (all true) with proper shape
            masks = torch::ones(batch_dims, torch::TensorOptions().dtype(torch::kBool).device(dirs.device()));
        }

        // Flatten batch dimensions for CUDA kernel
        auto dirs_flat = dirs.reshape({-1, 3});
        auto coeffs_flat = coeffs.reshape({-1, coeffs.size(-2), 3});
        auto masks_flat = masks.reshape({-1});

        // Call spherical harmonics forward - pass FULL coeffs!
        auto colors = gsplat::spherical_harmonics_fwd(
            sh_degree, dirs_flat, coeffs_flat, masks_flat);

        // Reshape output back to original batch dimensions
        auto output_shape = dirs.sizes().vec();
        output_shape[output_shape.size() - 1] = 3; // Ensure last dimension is 3
        colors = colors.reshape(output_shape).contiguous();

        TORCH_CHECK(colors.is_cuda(), "colors must be on CUDA after SH computation");

        // Save for backward - save everything as-is
        ctx->save_for_backward({dirs, coeffs, masks});
        ctx->saved_data["sh_degree"] = sh_degree;
        ctx->saved_data["num_bases"] = coeffs.size(-2); // Save the full K dimension

        return {colors};
    }

    torch::autograd::tensor_list SphericalHarmonicsFunction::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
        auto v_colors = grad_outputs[0].contiguous();

        auto saved = ctx->get_saved_variables();
        const auto& dirs = saved[0];
        const auto& coeffs = saved[1];
        const auto& masks = saved[2];

        const int sh_degree = ctx->saved_data["sh_degree"].to<int>();
        const int num_bases = ctx->saved_data["num_bases"].to<int>();

        // Flatten for CUDA kernel
        auto dirs_flat = dirs.reshape({-1, 3});
        auto coeffs_flat = coeffs.reshape({-1, num_bases, 3});
        auto masks_flat = masks.reshape({-1});
        auto v_colors_flat = v_colors.reshape({-1, 3});

        // Compute v_dirs based on needs_input_grad(1) (dirs is second input)
        bool compute_v_dirs = ctx->needs_input_grad(1);

        auto sh_grads = gsplat::spherical_harmonics_bwd(
            num_bases, sh_degree,
            dirs_flat, coeffs_flat, masks_flat,
            v_colors_flat, compute_v_dirs);

        auto v_coeffs = std::get<0>(sh_grads);
        auto v_dirs = std::get<1>(sh_grads);

        // Reshape gradients back to original shapes
        if (v_dirs.defined()) {
            v_dirs = v_dirs.reshape(dirs.sizes());
        }
        if (v_coeffs.defined()) {
            v_coeffs = v_coeffs.reshape(coeffs.sizes());
        }

        // Check gradient requirements
        if (!ctx->needs_input_grad(1)) {
            v_dirs = torch::Tensor();
        }
        if (!ctx->needs_input_grad(2)) {
            v_coeffs = torch::Tensor();
        }

        // Return gradients in same order as inputs: sh_degree_tensor, dirs, coeffs, masks
        return {torch::Tensor(), v_dirs, v_coeffs, torch::Tensor()};
    }

    // ProjectionFunction implementation
    torch::autograd::tensor_list fully_fused_projection_with_ut(
        torch::Tensor means3D,                          // [N, 3]
        torch::Tensor quats,                            // [N, 4]
        torch::Tensor scales,                           // [N, 3]
        torch::Tensor opacities,                        // [N]
        torch::Tensor viewmat,                          // [C, 4, 4]
        torch::Tensor K,                                // [C, 3, 3]
        std::optional<torch::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4]
        std::optional<torch::Tensor> tangential_coeffs, // [..., C, 2]
        std::optional<torch::Tensor> thin_prism_coeffs, // [..., C, 4]
        GUTProjectionSettings settings,
        UnscentedTransformParameters ut_params) {
        // Input validation
        const int N = static_cast<int>(means3D.size(0));
        const int C = static_cast<int>(viewmat.size(0));

        TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3,
                    "means3D must be [N, 3], got ", means3D.sizes());
        TORCH_CHECK(quats.dim() == 2 && quats.size(0) == N && quats.size(1) == 4,
                    "quats must be [N, 4], got ", quats.sizes());
        TORCH_CHECK(scales.dim() == 2 && scales.size(0) == N && scales.size(1) == 3,
                    "scales must be [N, 3], got ", scales.sizes());

        // Opacities is optional - only validate if defined
        if (opacities.defined()) {
            TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == N,
                        "opacities must be [N], got ", opacities.sizes());
        }

        TORCH_CHECK(viewmat.dim() == 3 && viewmat.size(1) == 4 && viewmat.size(2) == 4,
                    "viewmat must be [C, 4, 4], got ", viewmat.sizes());
        TORCH_CHECK(K.dim() == 3 && K.size(0) == C && K.size(1) == 3 && K.size(2) == 3,
                    "K must be [C, 3, 3], got ", K.sizes());
        if (radial_coeffs.has_value()) {
            TORCH_CHECK(radial_coeffs->size(-1) == 4 || radial_coeffs->size(-1) == 6,
                        "radial_coeffs last dimension must be 4 or 6, got ", radial_coeffs->size(-1));
        }
        if (tangential_coeffs.has_value()) {
            TORCH_CHECK(tangential_coeffs->size(-1) == 2,
                        "tangential_coeffs last dimension must be 2, got ", tangential_coeffs->size(-1));
        }
        if (thin_prism_coeffs.has_value()) {
            TORCH_CHECK(thin_prism_coeffs->size(-1) == 4,
                        "thin_prism_coeffs last dimension must be 4, got ", thin_prism_coeffs->size(-1));
        }

        // Device checks
        TORCH_CHECK(means3D.is_cuda(), "means3D must be on CUDA");
        TORCH_CHECK(quats.is_cuda(), "quats must be on CUDA");
        TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA");
        if (opacities.defined()) {
            TORCH_CHECK(opacities.is_cuda(), "opacities must be on CUDA");
        }
        TORCH_CHECK(viewmat.is_cuda(), "viewmat must be on CUDA");
        TORCH_CHECK(K.is_cuda(), "K must be on CUDA");
        if (radial_coeffs.has_value()) {
            TORCH_CHECK(radial_coeffs->is_cuda(), "radial_coeffs must be on CUDA");
            radial_coeffs = radial_coeffs->contiguous();
        }
        if (tangential_coeffs.has_value()) {
            TORCH_CHECK(tangential_coeffs->is_cuda(), "tangential_coeffs must be on CUDA");
            tangential_coeffs = tangential_coeffs->contiguous();
        }
        if (thin_prism_coeffs.has_value()) {
            TORCH_CHECK(thin_prism_coeffs->is_cuda(), "thin_prism_coeffs must be on CUDA");
            thin_prism_coeffs = thin_prism_coeffs->contiguous();
        }

        // Ensure all tensors are contiguous
        means3D = means3D.contiguous();
        quats = quats.contiguous();
        scales = scales.contiguous();
        if (opacities.defined()) {
            opacities = opacities.contiguous();
        }
        viewmat = viewmat.contiguous();
        K = K.contiguous();

        // Apply scaling modifier
        auto scaled_scales = scales * settings.scaling_modifier;

        // Call projection - pass undefined tensor if opacities not provided
        auto proj_results = gsplat::projection_ut_3dgs_fused(
            means3D,
            quats,
            scaled_scales,
            opacities,
            viewmat,
            std::nullopt,
            K,
            settings.width,
            settings.height,
            settings.eps2d,
            settings.near_plane,
            settings.far_plane,
            settings.radius_clip,
            false,
            settings.camera_model,
            ut_params,
            ShutterType::GLOBAL,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs);
        auto radii = std::get<0>(proj_results).contiguous();
        auto means2d = std::get<1>(proj_results).contiguous();
        auto depths = std::get<2>(proj_results).contiguous();
        auto conics = std::get<3>(proj_results).contiguous();
        auto compensations = std::get<4>(proj_results);

        if (!compensations.defined()) {
            compensations = at::empty({0});
        }

        // Validate outputs
        TORCH_CHECK(radii.dim() == 3 && radii.size(0) == C && radii.size(1) == N && radii.size(2) == 2,
                    "radii must be [C, N, 2], got ", radii.sizes());
        TORCH_CHECK(means2d.dim() == 3 && means2d.size(0) == C && means2d.size(1) == N && means2d.size(2) == 2,
                    "means2d must be [C, N, 2], got ", means2d.sizes());
        TORCH_CHECK(depths.dim() == 2 && depths.size(0) == C && depths.size(1) == N,
                    "depths must be [C, N], got ", depths.sizes());
        TORCH_CHECK(conics.dim() == 3 && conics.size(0) == C && conics.size(1) == N && conics.size(2) == 3,
                    "conics must be [C, N, 3], got ", conics.sizes());

        // Device checks for outputs
        TORCH_CHECK(radii.is_cuda(), "radii must be on CUDA");
        TORCH_CHECK(means2d.is_cuda(), "means2d must be on CUDA");
        TORCH_CHECK(depths.is_cuda(), "depths must be on CUDA");
        TORCH_CHECK(conics.is_cuda(), "conics must be on CUDA");

        return {radii, means2d, depths, conics, compensations};
    }

    torch::autograd::tensor_list GUTRasterizationFunction::forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor means3D,                          // [N, 3]
        torch::Tensor quats,                            // [N, 4]
        torch::Tensor scales,                           // [N, 3]
        torch::Tensor colors,                           // [N, C]
        torch::Tensor opacities,                        // [N]
        torch::Tensor bg_color,                         // [N, C]
        std::optional<torch::Tensor> masks,             // [N, C, tile_height, tile_width]
        torch::Tensor viewmat,                          // [C, 4, 4]
        torch::Tensor K,                                // [C, 3, 3]
        std::optional<torch::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4]
        std::optional<torch::Tensor> tangential_coeffs, // [..., C, 2]
        std::optional<torch::Tensor> thin_prism_coeffs, // [..., C, 4]
        torch::Tensor isect_offsets,                    // [C, tile_height, tile_width]
        torch::Tensor flatten_ids,                      // [nnz]
        GUTRasterizationSettings settings,
        UnscentedTransformParameters ut_params) {
        TORCH_CHECK(colors.size(-1) == 3, "Only 3 colors are supported currently.");
        ctx->saved_data["width"] = settings.width;
        ctx->saved_data["height"] = settings.height;
        ctx->saved_data["tile_size"] = settings.tile_size;
        ctx->saved_data["camera_model"] = settings.camera_model;
        ctx->saved_data["ut_params"] = ut_params.to_tensor();
        scales = scales * settings.scaling_modifier;
        auto results = gsplat::rasterize_to_pixels_from_world_3dgs_fwd(
            means3D.contiguous(),
            quats.contiguous(),
            scales.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            bg_color.contiguous(),
            masks.has_value() ? std::optional(masks->contiguous()) : std::nullopt,
            settings.width,
            settings.height,
            settings.tile_size,
            viewmat.contiguous(),
            std::nullopt,
            K.contiguous(),
            settings.camera_model,
            ut_params,
            ShutterType::GLOBAL,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            isect_offsets.contiguous(),
            flatten_ids.contiguous());

        auto render_colors = std::get<0>(results).contiguous();
        auto render_alpha = std::get<1>(results).contiguous();
        auto last_ids = std::get<2>(results).contiguous();

        ctx->save_for_backward({means3D, quats, scales, colors, opacities, bg_color,
                                masks.has_value() ? *masks : torch::Tensor(), viewmat, K,
                                radial_coeffs.has_value() ? *radial_coeffs : torch::Tensor(),
                                tangential_coeffs.has_value() ? *tangential_coeffs : torch::Tensor(),
                                thin_prism_coeffs.has_value() ? *thin_prism_coeffs : torch::Tensor(),
                                isect_offsets, flatten_ids, render_alpha, last_ids});

        return {render_colors, render_alpha};
    }

    torch::autograd::tensor_list GUTRasterizationFunction::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
        auto v_render_colors = grad_outputs[0].contiguous();
        auto v_render_alpha = grad_outputs[1].contiguous();

        auto saved = ctx->get_saved_variables();
        const auto& means3D = saved[0];
        const auto& quats = saved[1];
        const auto& scales = saved[2];
        const auto& colors = saved[3];
        const auto& opacities = saved[4];
        const auto& bg_color = saved[5];
        const std::optional<torch::Tensor> masks = saved[6].numel() > 0 ? std::optional(saved[6]) : std::nullopt;
        const auto& viewmat = saved[7];
        const auto& K = saved[8];
        const std::optional<torch::Tensor> radial_coeffs =
            saved[9].numel() > 0 ? std::optional(saved[9]) : std::nullopt;
        const std::optional<torch::Tensor> tangential_coeffs = saved[10].numel() > 0
                                                                   ? std::optional(saved[10])
                                                                   : std::nullopt;
        const std::optional<torch::Tensor> thin_prism_coeffs = saved[11].numel() > 0
                                                                   ? std::optional(saved[11])
                                                                   : std::nullopt;
        const auto& isect_offsets = saved[12];
        const auto& flatten_ids = saved[13];
        const auto& render_alpha = saved[14];
        const auto& last_ids = saved[15];

        // Extract settings
        const int width = ctx->saved_data["width"].toInt();
        const int height = ctx->saved_data["height"].toInt();
        const int tile_size = ctx->saved_data["tile_size"].toInt();
        const gsplat::CameraModelType camera_model =
            static_cast<gsplat::CameraModelType>(ctx->saved_data["camera_model"].toInt());
        auto ut_params = UnscentedTransformParameters::from_tensor(ctx->saved_data["ut_params"].toTensor());

        // Call backward
        auto raster_grads = gsplat::rasterize_to_pixels_from_world_3dgs_bwd(
            means3D, quats, scales, colors, opacities, bg_color, masks, width, height, tile_size,
            viewmat, std::nullopt, K, camera_model, ut_params, ShutterType::GLOBAL,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            isect_offsets, flatten_ids, render_alpha, last_ids,
            v_render_colors, v_render_alpha);

        // Extract gradients
        auto v_means3D = std::get<0>(raster_grads);
        auto v_quats = std::get<1>(raster_grads);
        auto v_scales = std::get<2>(raster_grads);
        auto v_colors = std::get<3>(raster_grads);
        auto v_opacities = std::get<4>(raster_grads);

        auto v_bg_color = torch::Tensor();
        if (ctx->needs_input_grad(5)) {
            v_bg_color = (v_render_colors * (1.0f - render_alpha)).toType(torch::kFloat32).sum({-3, -2});
        }

        return {
            v_means3D, v_quats, v_scales, v_colors, v_opacities,
            v_bg_color, torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
} // namespace gs::training
