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
            compensations = torch::ones({C, N},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
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
        TORCH_CHECK(compensations.is_cuda(), "compensations must be on CUDA");

        // Save for backward
        ctx->save_for_backward({means3D, quats, scaled_scales, opacities, viewmat, K, settings,
                                radii, conics, compensations});

        return {radii, means2d, depths, conics, compensations};
    }

    torch::autograd::tensor_list ProjectionFunction::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {

        auto v_radii = grad_outputs[0];
        auto v_means2d = grad_outputs[1].to(torch::kCUDA).contiguous();
        auto v_depths = grad_outputs[2].to(torch::kCUDA).contiguous();
        auto v_conics = grad_outputs[3].to(torch::kCUDA).contiguous();
        auto v_compensations = grad_outputs[4].to(torch::kCUDA).contiguous();

        auto saved = ctx->get_saved_variables();
        const auto& means3D = saved[0];
        const auto& quats = saved[1];
        const auto& scales = saved[2];
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

        // Call backward
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

        // v_opacities is computed from v_compensations only if opacities was defined
        torch::Tensor v_opacities;
        if (opacities.defined() && v_compensations.defined() && compensations.defined()) {
            v_opacities = (v_compensations * compensations / opacities.unsqueeze(0)).sum(0);
        } else {
            v_opacities = torch::Tensor(); // Return undefined tensor
        }

        return {v_means3D, v_quats, v_scales, v_opacities, v_viewmat, torch::Tensor(), torch::Tensor()};
    }

    // SphericalHarmonicsFunction implementation
    torch::autograd::tensor_list SphericalHarmonicsFunction::forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor sh_degree_tensor, // [1] containing sh_degree
        torch::Tensor dirs,             // [N, 3]
        torch::Tensor coeffs,           // [N, K, 3]
        torch::Tensor masks) {          // [N] optional

        const int sh_degree = sh_degree_tensor.item<int>();
        const int N = static_cast<int>(dirs.size(0));
        const int num_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);

        // Input validation
        TORCH_CHECK(dirs.dim() == 2 && dirs.size(1) == 3,
                    "dirs must be [N, 3], got ", dirs.sizes());
        TORCH_CHECK(coeffs.dim() == 3 && coeffs.size(0) == N && coeffs.size(2) == 3,
                    "coeffs must be [N, K, 3], got ", coeffs.sizes());
        TORCH_CHECK(coeffs.size(1) >= num_sh_coeffs,
                    "coeffs K dimension must be at least ", num_sh_coeffs, ", got ", coeffs.size(1));

        if (masks.defined()) {
            TORCH_CHECK(masks.dim() == 1 && masks.size(0) == N,
                        "masks must be [N], got ", masks.sizes());
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
            // Create default masks (all true)
            masks = torch::ones({N}, torch::TensorOptions().dtype(torch::kBool).device(dirs.device()));
        }

        // Use only the coefficients we need
        auto coeffs_used = coeffs.index({torch::indexing::Slice(),
                                         torch::indexing::Slice(torch::indexing::None, num_sh_coeffs),
                                         torch::indexing::Slice()})
                               .contiguous();

        // Call spherical harmonics forward
        auto colors = gsplat::spherical_harmonics_fwd(
            sh_degree, dirs, coeffs_used, masks);

        // Ensure colors is contiguous
        colors = colors.contiguous();
        TORCH_CHECK(colors.is_cuda(), "colors must be on CUDA after SH computation");

        // Save for backward
        ctx->save_for_backward({dirs, coeffs, coeffs_used, masks});
        ctx->saved_data["sh_degree"] = sh_degree;
        ctx->saved_data["num_sh_coeffs"] = num_sh_coeffs;

        return {colors};
    }

    torch::autograd::tensor_list SphericalHarmonicsFunction::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {

        auto v_colors = grad_outputs[0].to(torch::kCUDA).contiguous();

        auto saved = ctx->get_saved_variables();
        const auto& dirs = saved[0];
        const auto& coeffs = saved[1];
        const auto& coeffs_used = saved[2];
        const auto& masks = saved[3];

        const int sh_degree = ctx->saved_data["sh_degree"].to<int>();
        const int num_sh_coeffs = ctx->saved_data["num_sh_coeffs"].to<int>();

        // Compute v_dirs based on needs_input_grad[1] (dirs is second input)
        bool compute_v_dirs = ctx->needs_input_grad(1);

        auto sh_grads = gsplat::spherical_harmonics_bwd(
            num_sh_coeffs, sh_degree,
            dirs, coeffs_used, masks,
            v_colors, compute_v_dirs);

        auto v_coeffs_active = std::get<0>(sh_grads);
        auto v_dirs = std::get<1>(sh_grads);

        // Create full gradient tensor for coeffs
        torch::Tensor v_coeffs = torch::zeros_like(coeffs);
        v_coeffs.index_put_({torch::indexing::Slice(),
                             torch::indexing::Slice(torch::indexing::None, num_sh_coeffs),
                             torch::indexing::Slice()},
                            v_coeffs_active);

        if (!compute_v_dirs) {
            v_dirs = torch::Tensor();
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
        torch::Tensor bg_color,      // [C, channels] - may include depth
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
        // Remove the hardcoded channel check!
        TORCH_CHECK(colors.dim() == 3 && colors.size(0) == C && colors.size(1) == N,
                    "colors must be [C, N, channels], got ", colors.sizes());
        TORCH_CHECK(opacities.dim() == 2 && opacities.size(0) == C && opacities.size(1) == N,
                    "opacities must be [C, N], got ", opacities.sizes());
        TORCH_CHECK(bg_color.dim() == 2 && bg_color.size(0) == C && bg_color.size(1) == channels,
                    "bg_color must be [C, ", channels, "], got ", bg_color.sizes());

        // Device checks
        TORCH_CHECK(means2d.is_cuda(), "means2d must be on CUDA");
        TORCH_CHECK(conics.is_cuda(), "conics must be on CUDA");
        TORCH_CHECK(colors.is_cuda(), "colors must be on CUDA");
        TORCH_CHECK(opacities.is_cuda(), "opacities must be on CUDA");
        TORCH_CHECK(bg_color.is_cuda(), "bg_color must be on CUDA");
        TORCH_CHECK(isect_offsets.is_cuda(), "isect_offsets must be on CUDA");
        TORCH_CHECK(flatten_ids.is_cuda(), "flatten_ids must be on CUDA");
        TORCH_CHECK(settings.is_cuda(), "settings must be on CUDA");

        // Ensure tensors are contiguous
        means2d = means2d.contiguous();
        conics = conics.contiguous();
        colors = colors.contiguous();
        opacities = opacities.contiguous();
        bg_color = bg_color.contiguous();
        isect_offsets = isect_offsets.contiguous();
        flatten_ids = flatten_ids.contiguous();

        // Call rasterization
        auto raster_results = gsplat::rasterize_to_pixels_3dgs_fwd(
            means2d, conics, colors, opacities,
            bg_color, {}, // masks
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

        auto grad_image = grad_outputs[0].to(torch::kCUDA).contiguous();
        auto grad_alpha = grad_outputs[1].to(torch::kCUDA).contiguous();

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

        // Call backward
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

        // Background gradient
        torch::Tensor v_bg_color;
        if (bg_color.requires_grad()) {
            auto one_minus_alpha = 1.0f - rendered_alpha;
            v_bg_color = (grad_image * one_minus_alpha).sum({1, 2});
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

        return {v_quats, v_scales, torch::Tensor()};
    }

} // namespace gs