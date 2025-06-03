#include "core/rasterizer.hpp"
#include "Ops.h"
#include <torch/torch.h>

namespace gs {

    using namespace torch::indexing;

    // Autograd function for projection
    class ProjectionFunction : public torch::autograd::Function<ProjectionFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor means3D,    // [N, 3]
            torch::Tensor quats,      // [N, 4]
            torch::Tensor scales,     // [N, 3]
            torch::Tensor opacities,  // [N]
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
            TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == N,
                        "opacities must be [N], got ", opacities.sizes());
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
            TORCH_CHECK(opacities.is_cuda(), "opacities must be on CUDA");
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
            opacities = opacities.contiguous();
            viewmat = viewmat.contiguous();
            K = K.contiguous();

            // Apply scaling modifier
            auto scaled_scales = scales * scaling_modifier;

            // Call projection
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

        static torch::autograd::tensor_list backward(
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

            // v_opacities is computed from v_compensations
            torch::Tensor v_opacities;
            if (v_compensations.defined() && compensations.defined()) {
                v_opacities = (v_compensations * compensations / opacities.unsqueeze(0)).sum(0);
            } else {
                v_opacities = torch::zeros_like(opacities);
            }

            return {v_means3D, v_quats, v_scales, v_opacities, v_viewmat, torch::Tensor(), torch::Tensor()};
        }
    };

    // Autograd function for spherical harmonics
    class SphericalHarmonicsFunction : public torch::autograd::Function<SphericalHarmonicsFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor sh_coeffs,          // [N, K, 3]
            torch::Tensor means3D,            // [N, 3]
            torch::Tensor viewmat,            // [C, 4, 4]
            torch::Tensor radii,              // [C, N, 2]
            torch::Tensor sh_degree_tensor) { // [1] containing sh_degree

            const int N = static_cast<int>(means3D.size(0));
            const int C = static_cast<int>(viewmat.size(0));
            const int sh_degree = sh_degree_tensor.item<int>();

            // Input validation
            TORCH_CHECK(sh_coeffs.dim() == 3 && sh_coeffs.size(0) == N && sh_coeffs.size(2) == 3,
                        "sh_coeffs must be [N, K, 3], got ", sh_coeffs.sizes());
            TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3,
                        "means3D must be [N, 3], got ", means3D.sizes());
            TORCH_CHECK(viewmat.dim() == 3 && viewmat.size(1) == 4 && viewmat.size(2) == 4,
                        "viewmat must be [C, 4, 4], got ", viewmat.sizes());
            TORCH_CHECK(radii.dim() == 3 && radii.size(0) == C && radii.size(1) == N && radii.size(2) == 2,
                        "radii must be [C, N, 2], got ", radii.sizes());

            // Device checks before processing
            TORCH_CHECK(sh_coeffs.is_cuda(), "sh_coeffs must be on CUDA");
            TORCH_CHECK(means3D.is_cuda(), "means3D must be on CUDA");
            TORCH_CHECK(viewmat.is_cuda(), "viewmat must be on CUDA");
            TORCH_CHECK(radii.is_cuda(), "radii must be on CUDA");
            TORCH_CHECK(sh_degree_tensor.is_cuda(), "sh_degree_tensor must be on CUDA");

            // Ensure tensors are contiguous
            sh_coeffs = sh_coeffs.contiguous();
            means3D = means3D.contiguous();
            viewmat = viewmat.contiguous();
            radii = radii.contiguous();

            torch::Tensor colors;
            torch::Tensor sh_coeffs_used;
            int num_sh_coeffs = 1;

            if (sh_degree > 0 && sh_coeffs.size(1) > 1) {
                auto viewmat_inv = torch::inverse(viewmat);
                auto campos = viewmat_inv.index({Slice(), Slice(None, 3), 3});
                auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1);
                auto masks = (radii > 0).all(-1);

                // Device checks for intermediate tensors
                TORCH_CHECK(viewmat_inv.is_cuda(), "viewmat_inv must be on CUDA");
                TORCH_CHECK(campos.is_cuda(), "campos must be on CUDA");
                TORCH_CHECK(dirs.is_cuda(), "dirs must be on CUDA");
                TORCH_CHECK(masks.is_cuda(), "masks must be on CUDA");

                num_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
                sh_coeffs_used = sh_coeffs.index({Slice(), Slice(None, num_sh_coeffs), Slice()}).contiguous();

                // Compute SH for single camera
                colors = gsplat::spherical_harmonics_fwd(
                    sh_degree, dirs[0], sh_coeffs_used, masks[0]);
                colors = colors.unsqueeze(0);
                colors = torch::clamp_min(colors + 0.5f, 0.0f);
            } else {
                // Use only DC component
                sh_coeffs_used = sh_coeffs.index({Slice(), Slice(None, 1), Slice()}).contiguous();
                colors = sh_coeffs_used.index({Slice(), 0, Slice()});
                colors = colors.unsqueeze(0);
                colors = torch::clamp_min(colors + 0.5f, 0.0f);
            }

            // Ensure colors is on CUDA and contiguous
            colors = colors.contiguous();
            TORCH_CHECK(colors.is_cuda(), "colors must be on CUDA after SH computation");

            // Save for backward
            ctx->save_for_backward({sh_coeffs, sh_coeffs_used, means3D, viewmat, radii});
            ctx->saved_data["sh_degree"] = sh_degree;
            ctx->saved_data["num_sh_coeffs"] = num_sh_coeffs;

            return {colors};
        }

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {

            auto v_colors = grad_outputs[0].to(torch::kCUDA).contiguous();

            auto saved = ctx->get_saved_variables();
            const auto& sh_coeffs = saved[0];
            const auto& sh_coeffs_used = saved[1];
            const auto& means3D = saved[2];
            const auto& viewmat = saved[3];
            const auto& radii = saved[4];

            const int sh_degree = ctx->saved_data["sh_degree"].to<int>();
            const int num_sh_coeffs = ctx->saved_data["num_sh_coeffs"].to<int>();
            const int C = static_cast<int>(viewmat.size(0));
            const int N = static_cast<int>(means3D.size(0));

            torch::Tensor v_sh_coeffs = torch::zeros_like(sh_coeffs);
            torch::Tensor v_means3D;

            if (sh_degree > 0 && sh_coeffs.size(1) > 1) {
                auto viewmat_inv = torch::inverse(viewmat);
                auto campos = viewmat_inv.index({Slice(), Slice(None, 3), 3});
                auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1);
                auto masks = (radii > 0).all(-1);

                auto sh_grads = gsplat::spherical_harmonics_bwd(
                    num_sh_coeffs, sh_degree,
                    dirs[0], sh_coeffs_used, masks[0],
                    v_colors[0], true);

                auto v_sh_coeffs_active = std::get<0>(sh_grads);
                auto v_dirs = std::get<1>(sh_grads);

                v_sh_coeffs.index_put_({Slice(), Slice(None, num_sh_coeffs), Slice()}, v_sh_coeffs_active);
                v_means3D = v_dirs;
            } else {
                // Only DC component
                v_sh_coeffs.index_put_({Slice(), 0, Slice()}, v_colors[0]);
                v_means3D = torch::zeros_like(means3D);
            }

            return {v_sh_coeffs, v_means3D, torch::Tensor(), torch::Tensor(), torch::Tensor()};
        }
    };

    // Autograd function for rasterization
    class RasterizationFunction : public torch::autograd::Function<RasterizationFunction> {
    public:
        static torch::autograd::tensor_list forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor means2d,       // [C, N, 2]
            torch::Tensor conics,        // [C, N, 3]
            torch::Tensor colors,        // [C, N, 3]
            torch::Tensor opacities,     // [C, N]
            torch::Tensor bg_color,      // [C, 3]
            torch::Tensor isect_offsets, // [C, tile_height, tile_width]
            torch::Tensor flatten_ids,   // [nnz]
            torch::Tensor settings) {    // [3] containing width, height, tile_size

            // Extract settings
            const auto width = settings[0].item<int>();
            const auto height = settings[1].item<int>();
            const auto tile_size = settings[2].item<int>();

            const int C = static_cast<int>(means2d.size(0));
            const int N = static_cast<int>(means2d.size(1));

            // Input validation
            TORCH_CHECK(means2d.dim() == 3 && means2d.size(2) == 2,
                        "means2d must be [C, N, 2], got ", means2d.sizes());
            TORCH_CHECK(conics.dim() == 3 && conics.size(0) == C && conics.size(1) == N && conics.size(2) == 3,
                        "conics must be [C, N, 3], got ", conics.sizes());
            TORCH_CHECK(colors.dim() == 3 && colors.size(0) == C && colors.size(1) == N && colors.size(2) == 3,
                        "colors must be [C, N, 3], got ", colors.sizes());
            TORCH_CHECK(opacities.dim() == 2 && opacities.size(0) == C && opacities.size(1) == N,
                        "opacities must be [C, N], got ", opacities.sizes());
            TORCH_CHECK(bg_color.dim() == 2 && bg_color.size(0) == C && bg_color.size(1) == 3,
                        "bg_color must be [C, 3], got ", bg_color.sizes());

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

            // Validate outputs
            TORCH_CHECK(rendered_image.dim() == 4 && rendered_image.size(0) == C &&
                            rendered_image.size(1) == height && rendered_image.size(2) == width && rendered_image.size(3) == 3,
                        "rendered_image must be [C, H, W, 3], got ", rendered_image.sizes());
            TORCH_CHECK(rendered_alpha.dim() == 4 && rendered_alpha.size(0) == C &&
                            rendered_alpha.size(1) == height && rendered_alpha.size(2) == width && rendered_alpha.size(3) == 1,
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

        static torch::autograd::tensor_list backward(
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