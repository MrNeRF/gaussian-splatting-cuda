#include "core/rasterizer_autograd.hpp"

namespace gs {

    using namespace torch::indexing;

    // ProjectionFunction implementation
    torch::autograd::tensor_list ProjectionFunction::forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor means3D,    // [N, 3]
        torch::Tensor quats,      // [N, 4]
        torch::Tensor scales,     // [N, 3]
        torch::Tensor opacities,  // [N] or undefined (optional)
        torch::Tensor viewmat,    // [C, 4, 4]
        torch::Tensor K,          // [C, 3, 3]
        torch::Tensor settings) { // [7] tensor containing projection settings

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
        TORCH_CHECK(settings.dim() == 1 && settings.size(0) == 7,
                    "settings must be [7], got ", settings.sizes());

        // Device checks
        TORCH_CHECK(means3D.is_cuda(), "means3D must be on CUDA");
        TORCH_CHECK(quats.is_cuda(), "quats must be on CUDA");
        TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA");
        if (opacities.defined()) {
            TORCH_CHECK(opacities.is_cuda(), "opacities must be on CUDA");
        }
        TORCH_CHECK(viewmat.is_cuda(), "viewmat must be on CUDA");
        TORCH_CHECK(K.is_cuda(), "K must be on CUDA");
        TORCH_CHECK(settings.is_cuda(), "settings must be on CUDA");

        // Extract settings - keep on same device as input
        const auto width = settings[0].item<int>();
        const auto height = settings[1].item<int>();
        const auto eps2d = settings[2].item<float>();
        const auto near_plane = settings[3].item<float>();
        const auto far_plane = settings[4].item<float>();
        const auto radius_clip = settings[5].item<float>();
        const auto scaling_modifier = settings[6].item<float>();

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
        auto scaled_scales = scales * scaling_modifier;

        // Call projection - pass undefined tensor if opacities not provided
        auto proj_results = gsplat::projection_ewa_3dgs_fused_fwd(
            means3D,
            {}, // covars
            quats,
            scaled_scales,
            opacities, // Pass as-is (might be undefined)
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

        // Save for backward
        ctx->save_for_backward({means3D, quats, scaled_scales, opacities, viewmat, K, settings,
                                radii, conics, compensations});

        return {radii, means2d, depths, conics, compensations};
    }

    torch::autograd::tensor_list ProjectionFunction::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {

        auto v_radii = grad_outputs[0];
        auto v_means2d = grad_outputs[1].contiguous();
        auto v_depths = grad_outputs[2].contiguous();
        auto v_conics = grad_outputs[3].contiguous();
        auto v_compensations_tensor = grad_outputs[4];

        auto saved = ctx->get_saved_variables();
        const auto& means3D = saved[0];
        const auto& quats = saved[1];
        const auto& scaled_scales = saved[2]; // Note: this is already scaled!
        const auto& opacities = saved[3];
        const auto& viewmat = saved[4];
        const auto& K = saved[5];
        const auto& settings = saved[6];
        const auto& radii = saved[7];
        const auto& conics = saved[8];
        const auto& compensations = saved[9];

        // Extract settings
        const auto width = settings[0].item<int>();
        const auto height = settings[1].item<int>();
        const auto eps2d = settings[2].item<float>();
        const auto scaling_modifier = settings[6].item<float>();

        // Convert v_compensations to optional
        c10::optional<at::Tensor> v_compensations;
        if (v_compensations_tensor.defined() && v_compensations_tensor.numel() > 0) {
            v_compensations = v_compensations_tensor.to(torch::kCUDA).contiguous();
        }

        // Convert compensations to optional for backward call
        c10::optional<at::Tensor> compensations_opt;
        if (compensations.defined() && compensations.numel() > 0) {
            compensations_opt = compensations;
        }

        // Call backward - use scaled_scales here!
        auto proj_grads = gsplat::projection_ewa_3dgs_fused_bwd(
            means3D,
            {}, // covars
            quats,
            scaled_scales,
            viewmat,
            K,
            width,
            height,
            eps2d,
            gsplat::CameraModelType::PINHOLE,
            radii,
            conics,
            compensations_opt,
            v_means2d,
            v_depths,
            v_conics,
            v_compensations,
            ctx->needs_input_grad(4));

        auto v_means3D = std::get<0>(proj_grads);
        auto v_quats = std::get<2>(proj_grads);
        auto v_scales = std::get<3>(proj_grads); // This is gradient w.r.t. scaled_scales
        auto v_viewmat = std::get<4>(proj_grads);

        // v_scales is gradient w.r.t. scaled_scales, but we need gradient w.r.t. original scales
        // Since scaled_scales = scales * scaling_modifier, by chain rule:
        // d/d(scales) = d/d(scaled_scales) * scaling_modifier
        if (v_scales.defined()) {
            v_scales = v_scales * scaling_modifier;
        }

        // v_opacities is computed from v_compensations only if opacities was defined
        torch::Tensor v_opacities;
        if (opacities.defined() && v_compensations.has_value() && compensations_opt.has_value()) {
            v_opacities = (v_compensations.value() * compensations_opt.value() / opacities.unsqueeze(0)).sum(0);
        }

        // Check which inputs need gradients and set to undefined if not needed
        // Input order: means3D(0), quats(1), scales(2), opacities(3), viewmat(4), K(5), settings(6)
        if (!ctx->needs_input_grad(0)) { // means3D
            v_means3D = torch::Tensor();
        }
        if (!ctx->needs_input_grad(1)) { // quats
            v_quats = torch::Tensor();
        }
        if (!ctx->needs_input_grad(2)) { // scales
            v_scales = torch::Tensor();
        }
        if (!ctx->needs_input_grad(3)) { // opacities
            v_opacities = torch::Tensor();
        }
        if (!ctx->needs_input_grad(4)) { // viewmat
            v_viewmat = torch::Tensor();
        }

        // Return undefined tensors for K and settings (they don't have gradients)
        return {v_means3D, v_quats, v_scales, v_opacities, v_viewmat, torch::Tensor(), torch::Tensor()};
    }

    // SphericalHarmonicsFunction implementation
    torch::autograd::tensor_list SphericalHarmonicsFunction::forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor sh_degree_tensor, // [1] containing sh_degree
        torch::Tensor dirs,             // [..., 3]
        torch::Tensor coeffs,           // [..., K, 3]
        torch::Tensor masks) {          // [...] optional

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

    // RasterizationFunction implementation
    torch::autograd::tensor_list RasterizationFunction::forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor means2d,       // [C, N, 2]
        torch::Tensor conics,        // [C, N, 3]
        torch::Tensor colors,        // [C, N, channels] - may include depth
        torch::Tensor opacities,     // [C, N]
        torch::Tensor bg_color,      // [C, channels] - may include depth, can be empty
        torch::Tensor isect_offsets, // [C, tile_height, tile_width]
        torch::Tensor flatten_ids,   // [nnz]
        torch::Tensor settings) {    // [3] containing width, height, tile_size

        // Extract settings
        const auto width = settings[0].item<int>();
        const auto height = settings[1].item<int>();
        const auto tile_size = settings[2].item<int>();

        const int C = static_cast<int>(means2d.size(0));
        const int N = static_cast<int>(means2d.size(1));
        const int channels = static_cast<int>(colors.size(2)); // Get actual channel count

        // Input validation - DO NOT hardcode channels to 3!
        TORCH_CHECK(means2d.dim() == 3 && means2d.size(2) == 2,
                    "means2d must be [C, N, 2], got ", means2d.sizes());
        TORCH_CHECK(conics.dim() == 3 && conics.size(0) == C && conics.size(1) == N && conics.size(2) == 3,
                    "conics must be [C, N, 3], got ", conics.sizes());
        TORCH_CHECK(colors.dim() == 3 && colors.size(0) == C && colors.size(1) == N,
                    "colors must be [C, N, channels], got ", colors.sizes());
        TORCH_CHECK(opacities.dim() == 2 && opacities.size(0) == C && opacities.size(1) == N,
                    "opacities must be [C, N], got ", opacities.sizes());

        // Only validate bg_color if it's not empty
        if (bg_color.defined() && bg_color.numel() > 0) {
            TORCH_CHECK(bg_color.dim() == 2 && bg_color.size(0) == C && bg_color.size(1) == channels,
                        "bg_color must be [C, ", channels, "], got ", bg_color.sizes());
            TORCH_CHECK(bg_color.is_cuda(), "bg_color must be on CUDA");
            bg_color = bg_color.contiguous();
        }

        // Device checks
        TORCH_CHECK(means2d.is_cuda(), "means2d must be on CUDA");
        TORCH_CHECK(conics.is_cuda(), "conics must be on CUDA");
        TORCH_CHECK(colors.is_cuda(), "colors must be on CUDA");
        TORCH_CHECK(opacities.is_cuda(), "opacities must be on CUDA");
        TORCH_CHECK(isect_offsets.is_cuda(), "isect_offsets must be on CUDA");
        TORCH_CHECK(flatten_ids.is_cuda(), "flatten_ids must be on CUDA");
        TORCH_CHECK(settings.is_cuda(), "settings must be on CUDA");

        // Ensure tensors are contiguous
        means2d = means2d.contiguous();
        conics = conics.contiguous();
        colors = colors.contiguous();
        opacities = opacities.contiguous();
        isect_offsets = isect_offsets.contiguous();
        flatten_ids = flatten_ids.contiguous();

        // Convert empty tensor to optional for CUDA function
        at::optional<at::Tensor> bg_color_opt;
        if (bg_color.defined() && bg_color.numel() > 0) {
            bg_color_opt = bg_color;
        }
        // else bg_color_opt remains empty optional

        // Call rasterization with optional background
        auto raster_results = gsplat::rasterize_to_pixels_3dgs_fwd(
            means2d, conics, colors, opacities,
            bg_color_opt, {}, // bg_color_opt might not have value, masks is empty optional
            width, height, tile_size,
            isect_offsets, flatten_ids);

        auto rendered_image = std::get<0>(raster_results).contiguous();
        auto rendered_alpha = std::get<1>(raster_results).to(torch::kFloat32).contiguous();
        auto last_ids = std::get<2>(raster_results).contiguous();

        // Validate outputs - use actual channel count
        TORCH_CHECK(rendered_image.dim() == 4 && rendered_image.size(0) == C &&
                        rendered_image.size(1) == height && rendered_image.size(2) == width &&
                        rendered_image.size(3) == channels,
                    "rendered_image must be [C, H, W, ", channels, "], got ", rendered_image.sizes());
        TORCH_CHECK(rendered_alpha.dim() == 4 && rendered_alpha.size(0) == C &&
                        rendered_alpha.size(1) == height && rendered_alpha.size(2) == width &&
                        rendered_alpha.size(3) == 1,
                    "rendered_alpha must be [C, H, W, 1], got ", rendered_alpha.sizes());

        // Device checks for outputs
        TORCH_CHECK(rendered_image.is_cuda(), "rendered_image must be on CUDA");
        TORCH_CHECK(rendered_alpha.is_cuda(), "rendered_alpha must be on CUDA");
        TORCH_CHECK(last_ids.is_cuda(), "last_ids must be on CUDA");

        // Save for backward
        ctx->save_for_backward({means2d, conics, colors, opacities, bg_color,
                                isect_offsets, flatten_ids, rendered_alpha, last_ids, settings});

        return {rendered_image, rendered_alpha, last_ids};
    }

    torch::autograd::tensor_list RasterizationFunction::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {

        auto grad_image = grad_outputs[0].contiguous();
        auto grad_alpha = grad_outputs[1].contiguous();

        auto saved = ctx->get_saved_variables();
        const auto& means2d = saved[0];
        const auto& conics = saved[1];
        const auto& colors = saved[2];
        const auto& opacities = saved[3];
        const auto& bg_color = saved[4];
        const auto& isect_offsets = saved[5];
        const auto& flatten_ids = saved[6];
        const auto& rendered_alpha = saved[7];
        const auto& last_ids = saved[8];
        const auto& settings = saved[9];

        // Extract settings
        const auto width = settings[0].item<int>();
        const auto height = settings[1].item<int>();
        const auto tile_size = settings[2].item<int>();

        // Convert empty tensor to optional for CUDA function
        at::optional<at::Tensor> bg_color_opt;
        if (bg_color.defined() && bg_color.numel() > 0) {
            bg_color_opt = bg_color;
        }

        // Call backward
        auto raster_grads = gsplat::rasterize_to_pixels_3dgs_bwd(
            means2d, conics, colors, opacities,
            bg_color_opt, {}, // bg_color_opt might not have value, masks is empty optional
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

        // Background gradient - only compute if bg_color was not empty and needs gradient
        torch::Tensor v_bg_color;
        if (ctx->needs_input_grad(4) && bg_color.defined() && bg_color.numel() > 0) {
            auto one_minus_alpha = 1.0f - rendered_alpha;
            v_bg_color = (grad_image * one_minus_alpha).sum({1, 2});
        } else {
            v_bg_color = torch::Tensor();
        }

        // Check gradient requirements for other inputs
        if (!ctx->needs_input_grad(0)) {
            v_means2d = torch::Tensor();
        }
        if (!ctx->needs_input_grad(1)) {
            v_conics = torch::Tensor();
        }
        if (!ctx->needs_input_grad(2)) {
            v_colors = torch::Tensor();
        }
        if (!ctx->needs_input_grad(3)) {
            v_opacities = torch::Tensor();
        }

        return {v_means2d, v_conics, v_colors, v_opacities, v_bg_color,
                torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }

    // QuatScaleToCovarPreciFunction implementation
    torch::autograd::tensor_list QuatScaleToCovarPreciFunction::forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor quats,
        torch::Tensor scales,
        torch::Tensor settings) { // [3] tensor containing [compute_covar, compute_preci, triu]

        // Ensure inputs are contiguous and on CUDA
        quats = quats.contiguous();
        scales = scales.contiguous();

        TORCH_CHECK(quats.is_cuda(), "quats must be on CUDA");
        TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA");
        TORCH_CHECK(settings.is_cuda(), "settings must be on CUDA");

        // Extract settings
        bool compute_covar = settings[0].item<bool>();
        bool compute_preci = settings[1].item<bool>();
        bool triu = settings[2].item<bool>();

        auto [covars, precis] = gsplat::quat_scale_to_covar_preci_fwd(
            quats, scales, compute_covar, compute_preci, triu);

        // Ensure outputs are defined
        if (!covars.defined()) {
            if (triu) {
                covars = torch::zeros({quats.size(0), 6}, quats.options());
            } else {
                covars = torch::zeros({quats.size(0), 3, 3}, quats.options());
            }
        }
        if (!precis.defined()) {
            if (triu) {
                precis = torch::zeros({quats.size(0), 6}, quats.options());
            } else {
                precis = torch::zeros({quats.size(0), 3, 3}, quats.options());
            }
        }

        // Save for backward
        ctx->save_for_backward({quats, scales, settings, covars, precis});

        return {covars, precis};
    }

    torch::autograd::tensor_list QuatScaleToCovarPreciFunction::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {

        auto saved = ctx->get_saved_variables();
        auto quats = saved[0];
        auto scales = saved[1];
        auto settings = saved[2];

        bool triu = settings[2].item<bool>();

        auto v_covars = grad_outputs[0];
        auto v_precis = grad_outputs[1];

        // Ensure gradients are contiguous and on CUDA
        if (v_covars.defined()) {
            v_covars = v_covars.to(torch::kCUDA).contiguous();
        }
        if (v_precis.defined()) {
            v_precis = v_precis.to(torch::kCUDA).contiguous();
        }

        torch::Tensor v_quats, v_scales;

        if ((v_covars.defined() && v_covars.abs().sum().item<float>() > 0) ||
            (v_precis.defined() && v_precis.abs().sum().item<float>() > 0)) {
            auto [grad_quats, grad_scales] = gsplat::quat_scale_to_covar_preci_bwd(
                quats, scales, triu, v_covars, v_precis);
            v_quats = grad_quats;
            v_scales = grad_scales;
        } else {
            v_quats = torch::zeros_like(quats);
            v_scales = torch::zeros_like(scales);
        }

        // Check gradient requirements
        // Input order: quats(0), scales(1), settings(2)
        if (!ctx->needs_input_grad(0)) {
            v_quats = torch::Tensor();
        }
        if (!ctx->needs_input_grad(1)) {
            v_scales = torch::Tensor();
        }

        return {v_quats, v_scales, torch::Tensor()};
    }

} // namespace gs