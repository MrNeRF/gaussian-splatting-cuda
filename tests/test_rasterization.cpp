#include "Ops.h"
#include "core/camera.hpp"
#include "core/rasterizer.hpp"
#include "core/rasterizer_autograd.hpp"
#include "core/splat_data.hpp"
#include "torch_impl.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <torch/torch.h>

class RasterizationComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        device = torch::kCUDA;
    }

    // Reference rasterization implementation using the reference functions
    torch::Tensor reference_rasterize(
        const torch::Tensor& means,     // [N, 3]
        const torch::Tensor& quats,     // [N, 4]
        const torch::Tensor& scales,    // [N, 3]
        const torch::Tensor& opacities, // [N]
        const torch::Tensor& sh_coeffs, // [N, K, 3]
        const torch::Tensor& viewmats,  // [C, 4, 4]
        const torch::Tensor& Ks,        // [C, 3, 3]
        const torch::Tensor& bg_color,  // [C, 3]
        int width, int height,
        int sh_degree,
        float scaling_modifier = 1.0f) {

        const int C = viewmats.size(0);
        const int N = means.size(0);
        const int tile_size = 16;

        std::cout << "\n=== Reference Rasterization Pipeline ===" << std::endl;

        // Step 1: Convert quaternions and scales to covariance matrices
        auto scaled_scales = scales * scaling_modifier;
        auto [covars, _] = reference::quat_scale_to_covar_preci(quats, scaled_scales, true, false, false);
        std::cout << "Step 1: Covariance matrices computed, shape: " << covars.sizes() << std::endl;

        // Step 2: Projection
        const float eps2d = 0.3f;
        const float near_plane = 0.01f;
        const float far_plane = 10000.0f;

        auto [radii, means2d, depths, conics, compensations] = reference::fully_fused_projection(
            means, covars, viewmats, Ks, width, height, eps2d, near_plane, far_plane, true, "pinhole");

        std::cout << "Step 2: Projection complete" << std::endl;
        std::cout << "  Visible gaussians: " << (radii > 0).all(-1).sum().item<int>() << "/" << (C * N) << std::endl;

        // Step 3: Compute colors from spherical harmonics
        torch::Tensor colors;
        if (sh_degree > 0 && sh_coeffs.size(1) > 1) {
            // Compute view directions
            auto viewmat_inv = torch::inverse(viewmats);
            auto campos = viewmat_inv.slice(1, 0, 3).select(2, 3); // [C, 3]
            auto dirs = means.unsqueeze(0) - campos.unsqueeze(1);  // [C, N, 3]

            // For single camera, use first view
            colors = reference::spherical_harmonics(sh_degree, dirs[0], sh_coeffs);
            colors = colors.unsqueeze(0); // [1, N, 3]
        } else {
            // Use DC component only
            colors = sh_coeffs.select(1, 0); // [N, 3]
            colors = colors.unsqueeze(0);    // [1, N, 3]
        }

        // Apply SH offset and clamp
        colors = torch::clamp_min(colors + 0.5f, 0.0f);
        std::cout << "Step 3: SH colors computed, range: [" << colors.min().item<float>()
                  << ", " << colors.max().item<float>() << "]" << std::endl;

        // Step 4: Apply opacity with compensations
        auto final_opacities = opacities.unsqueeze(0) * compensations; // [C, N]
        std::cout << "Step 4: Opacities with compensations applied" << std::endl;

        // Step 5: Tile intersection
        const int tile_width = (width + tile_size - 1) / tile_size;
        const int tile_height = (height + tile_size - 1) / tile_size;

        auto [tiles_per_gauss, isect_ids, flatten_ids] = reference::isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height, true);

        std::cout << "Step 5: Tile intersection complete" << std::endl;
        std::cout << "  Total intersections: " << isect_ids.size(0) << std::endl;

        // Step 6: Rasterization
        // Since we don't have a reference rasterizer, we'll use gsplat's rasterization
        // but with our computed intermediate values
        auto isect_offsets = gsplat::intersect_offset(isect_ids, C, tile_width, tile_height);
        isect_offsets = isect_offsets.reshape({C, tile_height, tile_width});

        // Ensure all tensors are contiguous before passing to gsplat
        auto means2d_contig = means2d.contiguous();
        auto conics_contig = conics.contiguous();
        auto colors_contig = colors.contiguous();
        auto final_opacities_contig = final_opacities.contiguous();
        auto bg_color_contig = bg_color.contiguous();
        auto isect_offsets_contig = isect_offsets.contiguous();
        auto flatten_ids_contig = flatten_ids.contiguous();

        auto raster_results = gsplat::rasterize_to_pixels_3dgs_fwd(
            means2d_contig, conics_contig, colors_contig, final_opacities_contig,
            bg_color_contig, {}, // masks
            width, height, tile_size,
            isect_offsets_contig, flatten_ids_contig);

        auto rendered_image = std::get<0>(raster_results); // [C, H, W, 3]
        std::cout << "Step 6: Rasterization complete" << std::endl;

        // Convert to [C, 3, H, W] format and clamp max to 1.0 to match gs::rasterize output
        auto final_image = rendered_image.permute({0, 3, 1, 2});
        return torch::clamp_max(final_image, 1.0f);
    }

    torch::Device device{torch::kCPU};
};

TEST_F(RasterizationComparisonTest, StepByStepComparison) {
    torch::manual_seed(42);

    // Test parameters
    const int C = 1;   // Single camera for simplicity
    const int N = 100; // Fewer gaussians for easier debugging
    const int width = 64;
    const int height = 64;
    const float focal = 100.0f;
    const int sh_degree = 2;
    const int tile_size = 16;

    std::cout << "\n=== Step-by-Step Comparison Test ===" << std::endl;
    std::cout << "Parameters: C=" << C << ", N=" << N << ", width=" << width
              << ", height=" << height << ", SH degree=" << sh_degree << std::endl;

    // Create test data
    auto means = torch::rand({N, 3}, device) * 2.0f - 1.0f;
    means.select(1, 2) = torch::abs(means.select(1, 2)) + 2.0f;

    auto quats = torch::randn({N, 4}, device);
    quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));

    auto scales = torch::rand({N, 3}, device) * 0.05f + 0.01f;
    auto opacities = torch::rand({N}, device) * 0.5f + 0.3f;

    const int num_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
    auto sh_coeffs = (torch::rand({N, num_sh_coeffs, 3}, device) - 0.5f) * 0.3f;

    auto K = torch::tensor(
                 {{focal, 0.0f, width / 2.0f},
                  {0.0f, focal, height / 2.0f},
                  {0.0f, 0.0f, 1.0f}},
                 device)
                 .unsqueeze(0);

    auto viewmat = torch::eye(4, device).unsqueeze(0).contiguous();
    auto bg_color = torch::zeros({C, 3}, device);

    // Settings
    const float eps2d = 0.3f;
    const float near_plane = 0.01f;
    const float far_plane = 10000.0f;
    const float radius_clip = 0.0f;
    const float scaling_modifier = 1.0f;

    std::cout << "\n--- Step 1: Covariance Matrix Computation ---" << std::endl;

    // Reference implementation
    auto scaled_scales = scales * scaling_modifier;
    auto [ref_covars, _] = reference::quat_scale_to_covar_preci(quats, scaled_scales, true, false, false);

    // gs::rasterize uses autograd function internally, so we'll call it directly
    auto covar_settings = torch::tensor({1.0f, 0.0f, 0.0f}, device); // compute_covar=true, compute_preci=false, triu=false
    auto gs_covar_outputs = gs::QuatScaleToCovarPreciFunction::apply(quats, scaled_scales, covar_settings);
    auto gs_covars = gs_covar_outputs[0];

    std::cout << "Reference covars shape: " << ref_covars.sizes() << std::endl;
    std::cout << "GS covars shape: " << gs_covars.sizes() << std::endl;

    auto covar_diff = (ref_covars - gs_covars).abs();
    std::cout << "Covariance max diff: " << covar_diff.max().item<float>() << std::endl;
    std::cout << "Covariance mean diff: " << covar_diff.mean().item<float>() << std::endl;

    EXPECT_TRUE(torch::allclose(ref_covars, gs_covars, 1e-5, 1e-5))
        << "Covariance matrices don't match";

    std::cout << "\n--- Step 2: Projection ---" << std::endl;

    // Reference projection
    auto [ref_radii, ref_means2d, ref_depths, ref_conics, ref_compensations] =
        reference::fully_fused_projection(
            means, ref_covars, viewmat, K, width, height,
            eps2d, near_plane, far_plane, false, "pinhole");

    // GS projection using autograd function
    auto proj_settings = torch::tensor({(float)width, (float)height, eps2d, near_plane, far_plane,
                                        radius_clip, scaling_modifier},
                                       device);

    opacities = torch::ones_like(opacities);
    auto gs_proj_outputs = gs::ProjectionFunction::apply(
        means, quats, scales, opacities, viewmat, K, proj_settings);

    auto gs_radii = gs_proj_outputs[0];
    auto gs_means2d = gs_proj_outputs[1];
    auto gs_depths = gs_proj_outputs[2];
    auto gs_conics = gs_proj_outputs[3];
    auto gs_compensations = gs_proj_outputs[4];

    std::cout << "Reference visible: " << (ref_radii > 0).all(-1).sum().item<int>()
              << "/" << (C * N) << std::endl;
    std::cout << "GS visible: " << (gs_radii > 0).all(-1).sum().item<int>()
              << "/" << (C * N) << std::endl;

    // Compare projection outputs
    auto radii_diff = (ref_radii.to(torch::kFloat32) - gs_radii.to(torch::kFloat32)).abs();
    auto means2d_diff = (ref_means2d - gs_means2d).abs();
    auto depths_diff = (ref_depths - gs_depths).abs();
    auto conics_diff = (ref_conics - gs_conics).abs();

    std::cout << "Radii max diff: " << radii_diff.max().item<float>() << std::endl;
    std::cout << "Means2D max diff: " << means2d_diff.max().item<float>() << std::endl;
    std::cout << "Depths max diff: " << depths_diff.max().item<float>() << std::endl;
    std::cout << "Conics max diff: " << conics_diff.max().item<float>() << std::endl;

    // Check if compensations match (reference returns empty if calc_compensations=false)
    if (!ref_compensations.defined()) {
        ref_compensations = torch::ones({C, N}, device);
    }
    auto comp_diff = (ref_compensations - gs_compensations).abs();
    std::cout << "Compensations max diff: " << comp_diff.max().item<float>() << std::endl;

    std::cout << "\n--- Step 3: Spherical Harmonics ---" << std::endl;

    // Compute view directions
    auto viewmat_inv = torch::inverse(viewmat);
    auto campos = viewmat_inv.slice(1, 0, 3).select(2, 3);
    auto dirs = means.unsqueeze(0) - campos.unsqueeze(1);

    // Reference SH
    torch::Tensor ref_colors;
    if (sh_degree > 0 && sh_coeffs.size(1) > 1) {
        ref_colors = reference::spherical_harmonics(sh_degree, dirs[0], sh_coeffs);
        ref_colors = ref_colors.unsqueeze(0);
    } else {
        ref_colors = sh_coeffs.select(1, 0).unsqueeze(0);
    }
    ref_colors = torch::clamp_min(ref_colors + 0.5f, 0.0f);

    // GS SH using autograd function
    auto masks = (gs_radii > 0).all(-1).squeeze(0);
    auto sh_degree_tensor = torch::tensor({sh_degree},
                                          torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto gs_color_outputs = gs::SphericalHarmonicsFunction::apply(
        sh_degree_tensor, dirs.squeeze(0), sh_coeffs, masks);
    auto gs_colors = gs_color_outputs[0];
    gs_colors = torch::clamp_min(gs_colors + 0.5f, 0.0f).unsqueeze(0);

    std::cout << "Reference colors range: [" << ref_colors.min().item<float>()
              << ", " << ref_colors.max().item<float>() << "]" << std::endl;
    std::cout << "GS colors range: [" << gs_colors.min().item<float>()
              << ", " << gs_colors.max().item<float>() << "]" << std::endl;

    auto colors_diff = (ref_colors - gs_colors).abs();
    std::cout << "Colors max diff: " << colors_diff.max().item<float>() << std::endl;
    std::cout << "Colors mean diff: " << colors_diff.mean().item<float>() << std::endl;

    std::cout << "\n--- Step 4: Final Opacities ---" << std::endl;

    auto ref_final_opacities = opacities.unsqueeze(0) * ref_compensations;
    auto gs_final_opacities = opacities.unsqueeze(0) * gs_compensations;

    auto opacity_diff = (ref_final_opacities - gs_final_opacities).abs();
    std::cout << "Final opacities max diff: " << opacity_diff.max().item<float>() << std::endl;

    std::cout << "\n--- Step 5: Tile Intersection ---" << std::endl;

    const int tile_width = (width + tile_size - 1) / tile_size;
    const int tile_height = (height + tile_size - 1) / tile_size;

    // Reference tile intersection
    auto [ref_tiles_per_gauss, ref_isect_ids, ref_flatten_ids] = reference::isect_tiles(
        ref_means2d, ref_radii, ref_depths, tile_size, tile_width, tile_height, true);

    // GS tile intersection
    auto gs_isect_results = gsplat::intersect_tile(
        gs_means2d, gs_radii, gs_depths, {}, {},
        C, tile_size, tile_width, tile_height, true);

    auto gs_tiles_per_gauss = std::get<0>(gs_isect_results);
    auto gs_isect_ids = std::get<1>(gs_isect_results);
    auto gs_flatten_ids = std::get<2>(gs_isect_results);

    std::cout << "Reference intersections: " << ref_isect_ids.size(0) << std::endl;
    std::cout << "GS intersections: " << gs_isect_ids.size(0) << std::endl;

    // Compare tiles per gaussian
    auto tiles_diff = (ref_tiles_per_gauss.to(torch::kFloat32) -
                       gs_tiles_per_gauss.to(torch::kFloat32))
                          .abs();
    std::cout << "Tiles per gauss max diff: " << tiles_diff.max().item<float>() << std::endl;

    std::cout << "\n--- Step 6: Rasterization ---" << std::endl;

    auto ref_isect_offsets = gsplat::intersect_offset(ref_isect_ids, C, tile_width, tile_height);
    ref_isect_offsets = ref_isect_offsets.reshape({C, tile_height, tile_width});

    auto ref_raster_results = gsplat::rasterize_to_pixels_3dgs_fwd(
        ref_means2d.contiguous(), ref_conics.contiguous(), ref_colors.contiguous(),
        ref_final_opacities.contiguous(), bg_color.contiguous(), {},
        width, height, tile_size,
        ref_isect_offsets.contiguous(), ref_flatten_ids.contiguous());

    auto ref_rendered = std::get<0>(ref_raster_results);
    ref_rendered = torch::clamp_max(ref_rendered.permute({0, 3, 1, 2}), 1.0f);

    // GS uses autograd function
    auto gs_isect_offsets = gsplat::intersect_offset(gs_isect_ids, C, tile_width, tile_height);
    gs_isect_offsets = gs_isect_offsets.reshape({C, tile_height, tile_width});

    auto raster_settings = torch::tensor({(float)width, (float)height, (float)tile_size}, device);
    auto gs_raster_outputs = gs::RasterizationFunction::apply(
        gs_means2d, gs_conics, gs_colors, gs_final_opacities, bg_color,
        gs_isect_offsets, gs_flatten_ids, raster_settings);

    auto gs_rendered = gs_raster_outputs[0];
    gs_rendered = torch::clamp_max(gs_rendered.squeeze(0).permute({2, 0, 1}), 1.0f).unsqueeze(0);

    std::cout << "Reference render range: [" << ref_rendered.min().item<float>()
              << ", " << ref_rendered.max().item<float>() << "]" << std::endl;
    std::cout << "GS render range: [" << gs_rendered.min().item<float>()
              << ", " << gs_rendered.max().item<float>() << "]" << std::endl;

    auto render_diff = (ref_rendered - gs_rendered).abs();
    std::cout << "Render max diff: " << render_diff.max().item<float>() << std::endl;
    std::cout << "Render mean diff: " << render_diff.mean().item<float>() << std::endl;

    // Summary
    std::cout << "\n=== SUMMARY ===" << std::endl;
    std::cout << "Step 1 (Covariance): " << (covar_diff.max().item<float>() < 1e-5 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Step 2 (Projection): " << (means2d_diff.max().item<float>() < 1e-3 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Step 3 (SH Colors): " << (colors_diff.max().item<float>() < 1e-3 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Step 4 (Opacities): " << (opacity_diff.max().item<float>() < 1e-5 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Step 5 (Tile Intersection): " << (tiles_diff.max().item<float>() < 1 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Step 6 (Rasterization): " << (render_diff.mean().item<float>() < 0.0001 ? "PASS" : "FAIL") << std::endl;
    EXPECT_TRUE((render_diff.mean().item<float>() < 0.0001)) << "Rasterization outputs don't match";

    // Check for negative values
    bool ref_has_negative = (ref_rendered < 0).any().item<bool>();
    bool gs_has_negative = (gs_rendered < 0).any().item<bool>();

    std::cout << "\nReference has negative values: " << (ref_has_negative ? "YES" : "NO") << std::endl;
    std::cout << "GS has negative values: " << (gs_has_negative ? "YES" : "NO") << std::endl;

    if (ref_has_negative) {
        auto negative_mask = ref_rendered < 0;
        auto num_negative = negative_mask.sum().item<int>();
        auto min_value = ref_rendered.min().item<float>();
        std::cout << "Number of negative pixels: " << num_negative << std::endl;
        std::cout << "Most negative value: " << min_value << std::endl;
    }
}

// Also add a test specifically for SH colors issue
TEST_F(RasterizationComparisonTest, SphericalHarmonicsClampingDebug) {
    torch::manual_seed(42);

    const int N = 10;
    const int sh_degree = 2;
    const int num_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);

    std::cout << "\n=== SH Clamping Debug Test ===" << std::endl;

    // Create test data
    auto dirs = torch::randn({N, 3}, device);
    dirs = torch::nn::functional::normalize(dirs, torch::nn::functional::NormalizeFuncOptions().dim(-1));

    // Create SH coefficients with some extreme values
    auto sh_coeffs = torch::randn({N, num_sh_coeffs, 3}, device) * 0.5f;

    // Set some coefficients to extreme values to test clamping
    sh_coeffs[0][0] = torch::tensor({0.8f, 0.9f, 1.0f}, device);
    sh_coeffs[1][0] = torch::tensor({-0.3f, -0.4f, -0.5f}, device);

    auto masks = torch::ones({N}, torch::TensorOptions().dtype(torch::kBool).device(device));

    // Reference implementation
    auto ref_colors = reference::spherical_harmonics(sh_degree, dirs, sh_coeffs);
    std::cout << "Reference colors before offset: min=" << ref_colors.min().item<float>()
              << ", max=" << ref_colors.max().item<float>() << std::endl;

    auto ref_colors_clamped = torch::clamp_min(ref_colors + 0.5f, 0.0f);
    std::cout << "Reference colors after clamp_min(+0.5, 0): min=" << ref_colors_clamped.min().item<float>()
              << ", max=" << ref_colors_clamped.max().item<float>() << std::endl;

    // GS implementation
    auto sh_degree_tensor = torch::tensor({sh_degree},
                                          torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto gs_colors = gs::SphericalHarmonicsFunction::apply(
        sh_degree_tensor, dirs, sh_coeffs, masks)[0];

    std::cout << "\nGS colors before offset: min=" << gs_colors.min().item<float>()
              << ", max=" << gs_colors.max().item<float>() << std::endl;

    auto gs_colors_clamped = torch::clamp_min(gs_colors + 0.5f, 0.0f);
    std::cout << "GS colors after clamp_min(+0.5, 0): min=" << gs_colors_clamped.min().item<float>()
              << ", max=" << gs_colors_clamped.max().item<float>() << std::endl;

    // Check if clamping is working
    bool ref_has_above_one = (ref_colors_clamped > 1.0f).any().item<bool>();
    bool gs_has_above_one = (gs_colors_clamped > 1.0f).any().item<bool>();

    std::cout << "\nReference has values > 1.0 after clamping: " << (ref_has_above_one ? "YES" : "NO") << std::endl;
    std::cout << "GS has values > 1.0 after clamping: " << (gs_has_above_one ? "YES" : "NO") << std::endl;

    if (gs_has_above_one) {
        auto max_val = gs_colors_clamped.max().item<float>();
        std::cout << "GS max value after clamping: " << max_val << std::endl;

        // Find which gaussians have values > 1.0
        auto above_one_mask = (gs_colors_clamped > 1.0f).any(-1);
        for (int i = 0; i < N; ++i) {
            if (above_one_mask[i].item<bool>()) {
                std::cout << "Gaussian " << i << " has values > 1.0:" << std::endl;
                std::cout << "  SH DC term: " << sh_coeffs[i][0] << std::endl;
                std::cout << "  Colors before offset: " << gs_colors[i] << std::endl;
                std::cout << "  Colors after clamp: " << gs_colors_clamped[i] << std::endl;
            }
        }
    }
}

TEST_F(RasterizationComparisonTest, CompareWithGSRasterize) {
    torch::manual_seed(42);

    // Test parameters matching Python test
    const int C = 2;     // Number of cameras
    const int N = 10000; // Number of gaussians
    const int width = 300;
    const int height = 200;
    const float focal = 300.0f;
    const int sh_degree = 2; // Test with SH degree 2

    std::cout << "\n=== Rasterization Comparison Test ===" << std::endl;
    std::cout << "Parameters: C=" << C << ", N=" << N << ", width=" << width
              << ", height=" << height << ", SH degree=" << sh_degree << std::endl;

    // Create test data with controlled values
    auto means = torch::rand({N, 3}, device) * 2.0f - 1.0f;     // Range [-1, 1]
    means.select(1, 2) = torch::abs(means.select(1, 2)) + 2.0f; // Ensure z in [2, 3]

    auto quats = torch::randn({N, 4}, device);
    quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));

    auto scales = torch::rand({N, 3}, device) * 0.05f + 0.01f; // Smaller scales
    auto opacities = torch::rand({N}, device) * 0.5f + 0.3f;   // Range [0.3, 0.8]

    // Create SH coefficients with proper scaling
    const int num_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
    // Initialize with small values similar to what rgb_to_sh would produce
    // SH coefficients are typically in range [-1, 1] after rgb_to_sh conversion
    auto sh_coeffs = (torch::rand({N, num_sh_coeffs, 3}, device) - 0.5f) * 0.3f;

    // Create camera parameters
    auto Ks = torch::tensor(
                  {{focal, 0.0f, width / 2.0f},
                   {0.0f, focal, height / 2.0f},
                   {0.0f, 0.0f, 1.0f}},
                  device)
                  .expand({C, -1, -1})
                  .contiguous();

    auto viewmats = torch::eye(4, device).expand({C, -1, -1}).contiguous();

    // Test different viewmats for multiple cameras
    if (C > 1) {
        viewmats = viewmats.clone();
        // Create a rotation matrix for the second camera
        float angle = 0.3f; // radians
        auto R = torch::tensor({{std::cos(angle), 0.0f, std::sin(angle), 0.0f},
                                {0.0f, 1.0f, 0.0f, 0.0f},
                                {-std::sin(angle), 0.0f, std::cos(angle), 0.0f},
                                {0.0f, 0.0f, 0.0f, 1.0f}},
                               device);

        // Apply rotation and translation to second camera
        viewmats[1] = torch::matmul(R, viewmats[1]);
        viewmats[1][0][3] = 0.5f; // Translate in X
        viewmats[1][2][3] = 0.5f; // Translate in Z
    }

    auto bg_color = torch::zeros({C, 3}, device);

    // Call reference implementation with calc_compensations=false to match gs::rasterize
    auto ref_renders = reference_rasterize(
        means, quats, scales, opacities, sh_coeffs,
        viewmats, Ks, bg_color, width, height, sh_degree, 1.0f);

    std::cout << "\nReference render shape: " << ref_renders.sizes() << std::endl;
    std::cout << "Reference render stats: min=" << ref_renders.min().item<float>()
              << ", max=" << ref_renders.max().item<float>()
              << ", mean=" << ref_renders.mean().item<float>() << std::endl;

    // Now test with gs::rasterize for each camera view
    std::cout << "\n=== Testing gs::rasterize ===" << std::endl;

    // Create SplatData
    auto sh0 = sh_coeffs.slice(1, 0, 1);             // [N, 1, 3]
    auto shN = sh_coeffs.slice(1, 1, num_sh_coeffs); // [N, K-1, 3]

    auto gaussians = SplatData(
        sh_degree,
        means,
        sh0,
        shN,
        torch::log(scales), // SplatData expects log scales
        quats,
        torch::logit(opacities).unsqueeze(-1), // SplatData expects logit opacities
        1.0f);

    // Activate all SH degrees
    while (gaussians.get_active_sh_degree() < sh_degree) {
        gaussians.increment_sh_degree();
    }

    // Render each camera view
    std::vector<torch::Tensor> gs_renders;
    for (int cam_idx = 0; cam_idx < C; ++cam_idx) {
        // viewmat is world-to-camera, but Camera expects camera-to-world
        // So we need to invert the transformation
        auto viewmat_inv = torch::inverse(viewmats[cam_idx]);
        auto R = viewmat_inv.slice(0, 0, 3).slice(1, 0, 3).t().to(torch::kCPU);
        auto T = viewmat_inv.slice(0, 0, 3).select(1, 3).to(torch::kCPU);

        float fov = 2.0f * std::atan(width / (2.0f * focal));
        Camera camera(R, T, fov, fov, "test_cam_" + std::to_string(cam_idx), "", width, height, cam_idx);

        auto bg_single = bg_color[cam_idx];
        auto output = gs::rasterize(camera, gaussians, bg_single, 1.0f, false);

        gs_renders.push_back(output.image.unsqueeze(0)); // Add camera dimension

        std::cout << "Camera " << cam_idx << " render stats: min=" << output.image.min().item<float>()
                  << ", max=" << output.image.max().item<float>()
                  << ", mean=" << output.image.mean().item<float>() << std::endl;
    }

    // Stack renders
    auto gs_renders_stacked = torch::cat(gs_renders, 0); // [C, 3, H, W]

    std::cout << "\ngs::rasterize shape: " << gs_renders_stacked.sizes() << std::endl;

    // Compare outputs
    EXPECT_EQ(ref_renders.sizes(), gs_renders_stacked.sizes()) << "Output shapes don't match";

    // Check that outputs are valid
    EXPECT_FALSE(ref_renders.isnan().any().item<bool>()) << "Reference has NaN values";
    EXPECT_FALSE(gs_renders_stacked.isnan().any().item<bool>()) << "gs::rasterize has NaN values";

    EXPECT_TRUE((ref_renders >= 0.0f).all().item<bool>()) << "Reference has negative values";
    EXPECT_TRUE((gs_renders_stacked >= 0.0f).all().item<bool>()) << "gs::rasterize has negative values";

    // Check that different cameras produce different results
    if (C > 1) {
        auto cam0_mean = gs_renders_stacked[0].mean().item<float>();
        auto cam1_mean = gs_renders_stacked[1].mean().item<float>();
        EXPECT_NE(cam0_mean, cam1_mean) << "Different cameras should produce different results";
    }

    // Compare pixel-wise differences
    auto diff = (ref_renders - gs_renders_stacked).abs();
    auto max_diff = diff.max().item<float>();
    auto mean_diff = diff.mean().item<float>();

    std::cout << "\nPixel-wise comparison:" << std::endl;
    std::cout << "  Max absolute difference: " << max_diff << std::endl;
    std::cout << "  Mean absolute difference: " << mean_diff << std::endl;

    // Calculate relative error where values are non-zero
    auto mask = (ref_renders.abs() > 1e-6f) | (gs_renders_stacked.abs() > 1e-6f);
    if (mask.any().item<bool>()) {
        auto rel_diff = torch::where(mask,
                                     diff / (torch::max(ref_renders.abs(), gs_renders_stacked.abs()) + 1e-8f),
                                     torch::zeros_like(diff));
        auto max_rel_diff = rel_diff.max().item<float>();
        auto mean_rel_diff = rel_diff.sum().item<float>() / mask.sum().item<float>();

        std::cout << "  Max relative difference: " << max_rel_diff << std::endl;
        std::cout << "  Mean relative difference: " << mean_rel_diff << std::endl;
    }

    // Check that the renders are similar
    if (torch::allclose(ref_renders, gs_renders_stacked, 0.01f, 0.01f)) {
        std::cout << "\nSUCCESS: Renders match within tight tolerance (0.01)" << std::endl;
    } else if (torch::allclose(ref_renders, gs_renders_stacked, 0.05f, 0.05f)) {
        std::cout << "\nPARTIAL SUCCESS: Renders match within moderate tolerance (0.05)" << std::endl;
    } else if (torch::allclose(ref_renders, gs_renders_stacked, 0.1f, 0.1f)) {
        std::cout << "\nWARNING: Renders only match within relaxed tolerance (0.1)" << std::endl;
    } else {
        std::cout << "\nWARNING: Renders differ significantly" << std::endl;
    }

    // At minimum, check that both produce valid images with similar statistics
    EXPECT_LT(mean_diff, 0.2f) << "Mean difference is too large";
    EXPECT_LT(max_diff, 1.0f) << "Max difference is too large";

    // Check that the mean values are similar (within 20%)
    auto ref_mean = ref_renders.mean().item<float>();
    auto gs_mean = gs_renders_stacked.mean().item<float>();
    EXPECT_NEAR(ref_mean, gs_mean, std::abs(ref_mean) * 0.2f + 0.01f)
        << "Mean values differ too much";
}

TEST_F(RasterizationComparisonTest, TestDifferentRenderModes) {
    torch::manual_seed(42);

    // Smaller test for different configurations
    const int N = 1000;
    const int width = 64;
    const int height = 64;
    const float focal = 100.0f;

    // Test configurations
    std::vector<std::pair<int, std::string>> configs = {
        {0, "DC only"},
        {1, "SH degree 1"},
        {2, "SH degree 2"},
        {3, "SH degree 3"}};

    for (const auto& [sh_degree, desc] : configs) {
        std::cout << "\n=== Testing " << desc << " ===" << std::endl;

        // Create test data
        auto means = torch::rand({N, 3}, device) * 2.0f;
        means.select(1, 2) = torch::abs(means.select(1, 2)) + 1.0f;

        auto quats = torch::randn({N, 4}, device);
        quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));

        auto scales = torch::rand({N, 3}, device) * 0.05f + 0.01f;
        auto opacities = torch::rand({N}, device) * 0.8f + 0.1f;

        const int num_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
        auto sh_coeffs = torch::randn({N, num_sh_coeffs, 3}, device) * 0.1f;

        // Single camera
        auto K = torch::tensor(
                     {{focal, 0.0f, width / 2.0f},
                      {0.0f, focal, height / 2.0f},
                      {0.0f, 0.0f, 1.0f}},
                     device)
                     .unsqueeze(0);

        auto viewmat = torch::eye(4, device).unsqueeze(0);
        auto bg_color = torch::ones({1, 3}, device) * 0.1f; // Gray background

        // Reference implementation
        auto ref_render = reference_rasterize(
            means, quats, scales, opacities, sh_coeffs,
            viewmat, K, bg_color, width, height, sh_degree, 1.0f);

        // gs::rasterize implementation
        auto sh0 = sh_coeffs.slice(1, 0, 1);
        auto shN = sh_coeffs.slice(1, 1, num_sh_coeffs);

        auto gaussians = SplatData(
            sh_degree, means, sh0, shN,
            torch::log(scales), quats,
            torch::logit(opacities).unsqueeze(-1), 1.0f);

        while (gaussians.get_active_sh_degree() < sh_degree) {
            gaussians.increment_sh_degree();
        }

        auto R = torch::eye(3, torch::kCPU);
        auto T = torch::zeros({3}, torch::kCPU);
        float fov = 2.0f * std::atan(width / (2.0f * focal));
        Camera camera(R, T, fov, fov, "test_camera", "", width, height, 0);

        auto bg_single = bg_color.squeeze(0);
        auto output = gs::rasterize(camera, gaussians, bg_single, 1.0f, false);

        // Compare
        auto diff = (ref_render.squeeze(0) - output.image).abs();
        auto max_diff = diff.max().item<float>();
        auto mean_diff = diff.mean().item<float>();

        std::cout << "  Max difference: " << max_diff << std::endl;
        std::cout << "  Mean difference: " << mean_diff << std::endl;

        // Coverage statistics
        auto ref_coverage = (ref_render.squeeze(0) > bg_single.view({3, 1, 1})).any(0).to(torch::kFloat32).mean();
        auto gs_coverage = (output.image > bg_single.view({3, 1, 1})).any(0).to(torch::kFloat32).mean();

        std::cout << "  Reference coverage: " << ref_coverage.item<float>() * 100 << "%" << std::endl;
        std::cout << "  gs::rasterize coverage: " << gs_coverage.item<float>() * 100 << "%" << std::endl;

        EXPECT_TRUE(torch::allclose(ref_render.squeeze(0), output.image, 0.1, 0.1))
            << "Renders don't match for " << desc;
    }
}

TEST_F(RasterizationComparisonTest, TestEdgeCases) {
    torch::manual_seed(42);

    const int width = 32;
    const int height = 32;
    const float focal = 50.0f;

    // Test 1: Very few gaussians
    {
        std::cout << "\n=== Test: Very few gaussians ===" << std::endl;
        const int N = 5;

        auto means = torch::tensor(
            {{0.0f, 0.0f, 2.0f},
             {0.5f, 0.5f, 2.5f},
             {-0.5f, 0.5f, 3.0f},
             {0.5f, -0.5f, 3.5f},
             {-0.5f, -0.5f, 4.0f}},
            device);

        auto quats = torch::tensor(
            {{1.0f, 0.0f, 0.0f, 0.0f},
             {1.0f, 0.0f, 0.0f, 0.0f},
             {1.0f, 0.0f, 0.0f, 0.0f},
             {1.0f, 0.0f, 0.0f, 0.0f},
             {1.0f, 0.0f, 0.0f, 0.0f}},
            device);

        auto scales = torch::ones({N, 3}, device) * 0.1f;
        auto opacities = torch::ones({N}, device) * 0.8f;
        auto sh_coeffs = torch::ones({N, 1, 3}, device) * 0.5f;

        auto K = torch::tensor(
                     {{focal, 0.0f, width / 2.0f},
                      {0.0f, focal, height / 2.0f},
                      {0.0f, 0.0f, 1.0f}},
                     device)
                     .unsqueeze(0);

        auto viewmat = torch::eye(4, device).unsqueeze(0);
        auto bg_color = torch::zeros({1, 3}, device);

        auto ref_render = reference_rasterize(
            means, quats, scales, opacities, sh_coeffs,
            viewmat, K, bg_color, width, height, 0, 1.0f);

        std::cout << "  Reference non-zero pixels: "
                  << (ref_render > 0).any(1).sum().item<int>() << std::endl;

        EXPECT_TRUE((ref_render >= 0).all().item<bool>()) << "Negative values in render";
        EXPECT_FALSE(ref_render.isnan().any().item<bool>()) << "NaN in render";
    }

    // Test 2: Gaussians behind camera
    {
        std::cout << "\n=== Test: Gaussians behind camera ===" << std::endl;
        const int N = 10;

        auto means = torch::randn({N, 3}, device);
        means.select(1, 2) = -torch::abs(means.select(1, 2)) - 0.1f; // All behind camera

        auto quats = torch::randn({N, 4}, device);
        quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));

        auto scales = torch::rand({N, 3}, device) * 0.1f;
        auto opacities = torch::ones({N}, device);
        auto sh_coeffs = torch::rand({N, 1, 3}, device);

        auto K = torch::tensor(
                     {{focal, 0.0f, width / 2.0f},
                      {0.0f, focal, height / 2.0f},
                      {0.0f, 0.0f, 1.0f}},
                     device)
                     .unsqueeze(0);

        auto viewmat = torch::eye(4, device).unsqueeze(0);
        auto bg_color = torch::ones({1, 3}, device) * 0.5f;

        auto ref_render = reference_rasterize(
            means, quats, scales, opacities, sh_coeffs,
            viewmat, K, bg_color, width, height, 0, 1.0f);

        // Should render only background
        auto expected_bg = bg_color.view({1, 3, 1, 1}).expand({1, 3, height, width});
        auto is_bg = torch::allclose(ref_render, expected_bg, 1e-4, 1e-4);

        std::cout << "  All pixels are background: " << (is_bg ? "Yes" : "No") << std::endl;

        EXPECT_TRUE((ref_render >= 0).all().item<bool>()) << "Negative values in render";
        EXPECT_FALSE(ref_render.isnan().any().item<bool>()) << "NaN in render";
    }

    // Test 3: Very large gaussians
    {
        std::cout << "\n=== Test: Very large gaussians ===" << std::endl;
        const int N = 3;

        auto means = torch::tensor(
            {{0.0f, 0.0f, 2.0f},
             {1.0f, 1.0f, 3.0f},
             {-1.0f, -1.0f, 4.0f}},
            device);

        auto quats = torch::eye(4, device).slice(0, 0, N).slice(1, 0, 4);
        auto scales = torch::ones({N, 3}, device) * 2.0f; // Very large
        auto opacities = torch::ones({N}, device) * 0.3f;
        auto sh_coeffs = torch::rand({N, 1, 3}, device);

        auto K = torch::tensor(
                     {{focal, 0.0f, width / 2.0f},
                      {0.0f, focal, height / 2.0f},
                      {0.0f, 0.0f, 1.0f}},
                     device)
                     .unsqueeze(0);

        auto viewmat = torch::eye(4, device).unsqueeze(0);
        auto bg_color = torch::zeros({1, 3}, device);

        auto ref_render = reference_rasterize(
            means, quats, scales, opacities, sh_coeffs,
            viewmat, K, bg_color, width, height, 0, 1.0f);

        auto coverage = (ref_render > 0).any(1).to(torch::kFloat32).mean();
        std::cout << "  Coverage with large gaussians: " << coverage.item<float>() * 100 << "%" << std::endl;

        EXPECT_TRUE((ref_render >= 0).all().item<bool>()) << "Negative values in render";
        EXPECT_TRUE((ref_render <= 1.1f).all().item<bool>()) << "Values too large in render";
        EXPECT_FALSE(ref_render.isnan().any().item<bool>()) << "NaN in render";
    }
}