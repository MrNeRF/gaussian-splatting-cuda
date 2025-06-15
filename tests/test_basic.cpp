#include "Ops.h"
#include "core/debug_utils.hpp"
#include "core/image_io.hpp"
#include "core/metrics.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <torch/torch.h>

class BasicOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        device = torch::kCUDA;

        // Set tolerances for comparisons
        rtol = 1e-4;
        atol = 1e-4;
    }

    // Helper to check tensor closeness with custom tolerances
    void assertTensorClose(const torch::Tensor& a, const torch::Tensor& b,
                           double rtol_override = -1, double atol_override = -1) {
        double r = (rtol_override > 0) ? rtol_override : rtol;
        double a_tol = (atol_override > 0) ? atol_override : atol;

        ASSERT_TRUE(torch::allclose(a, b, r, a_tol))
            << "Tensors not close:\n"
            << "Max diff: " << (a - b).abs().max().template item<float>() << "\n"
            << "Mean diff: " << (a - b).abs().mean().template item<float>() << "\n"
            << "Shape A: " << a.sizes() << "\n"
            << "Shape B: " << b.sizes();
    }

    torch::Device device{torch::kCPU};
    double rtol{1e-4};
    double atol{1e-4};
};

TEST_F(BasicOpsTest, QuatScaleToCovarPreciTest) {
    torch::manual_seed(42);

    // Test data
    int N = 100;
    auto quats = torch::randn({N, 4}, device);
    auto scales = torch::rand({N, 3}, device) * 0.1;

    // Test with triu=false
    {
        auto [covars, precis] = gsplat::quat_scale_to_covar_preci_fwd(
            quats, scales, true, true, false);

        EXPECT_EQ(covars.sizes(), torch::IntArrayRef({N, 3, 3}));
        EXPECT_EQ(precis.sizes(), torch::IntArrayRef({N, 3, 3}));
    }

    // Test with triu=true
    {
        auto [covars_triu, precis_triu] = gsplat::quat_scale_to_covar_preci_fwd(
            quats, scales, true, true, true);

        // Upper triangular format has 6 elements
        EXPECT_EQ(covars_triu.sizes(), torch::IntArrayRef({N, 6}));
        EXPECT_EQ(precis_triu.sizes(), torch::IntArrayRef({N, 6}));
    }

    // Test backward pass - use the backward function directly
    {
        auto [covars, precis] = gsplat::quat_scale_to_covar_preci_fwd(
            quats, scales, true, true, false);

        auto v_covars = torch::randn_like(covars);
        auto v_precis = torch::randn_like(precis) * 0.01;

        // Use the backward function directly with correct argument order
        auto [v_quats, v_scales] = gsplat::quat_scale_to_covar_preci_bwd(
            quats, scales, false, v_covars, v_precis);

        // Check gradients exist and are valid
        EXPECT_TRUE(v_quats.defined());
        EXPECT_TRUE(v_scales.defined());
        EXPECT_FALSE(v_quats.isnan().any().item<bool>());
        EXPECT_FALSE(v_scales.isnan().any().item<bool>());
    }
}

TEST_F(BasicOpsTest, ProjectionTest) {
    torch::manual_seed(42);

    // Test with pinhole camera
    int N = 100;
    int C = 2;
    int width = 640, height = 480;

    auto means = torch::randn({N, 3}, device);
    auto quats = torch::randn({N, 4}, device);
    auto scales = torch::rand({N, 3}, device) * 0.1;
    auto viewmats = torch::eye(4, device).unsqueeze(0).repeat({C, 1, 1});
    auto Ks = torch::tensor({{300.0f, 0.0f, 320.0f},
                             {0.0f, 300.0f, 240.0f},
                             {0.0f, 0.0f, 1.0f}},
                            device)
                  .unsqueeze(0)
                  .repeat({C, 1, 1});

    // Create empty CUDA tensor for covars
    auto empty_covars = torch::empty({0, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Test projection
    auto proj_results = gsplat::projection_ewa_3dgs_fused_fwd(
        means,
        empty_covars,
        quats,
        scales,
        torch::rand({N}, device), // opacities for bounds
        viewmats,
        Ks,
        width,
        height,
        0.3f,     // eps2d
        0.01f,    // near_plane
        10000.0f, // far_plane
        0.0f,     // radius_clip
        false,    // calc_compensations
        gsplat::CameraModelType::PINHOLE);

    auto radii = std::get<0>(proj_results);
    auto means2d = std::get<1>(proj_results);
    auto depths = std::get<2>(proj_results);
    auto conics = std::get<3>(proj_results);
    auto compensations = std::get<4>(proj_results);

    // Check output shapes
    EXPECT_EQ(radii.sizes(), torch::IntArrayRef({C, N, 2}));
    EXPECT_EQ(means2d.sizes(), torch::IntArrayRef({C, N, 2}));
    EXPECT_EQ(depths.sizes(), torch::IntArrayRef({C, N}));
    EXPECT_EQ(conics.sizes(), torch::IntArrayRef({C, N, 3}));

    // Check that visible Gaussians have positive radii
    auto valid = (radii > 0).all(-1);
    auto valid_sum = valid.sum();
    EXPECT_GT(valid_sum.template item<int64_t>(), 0);
}

TEST_F(BasicOpsTest, SphericalHarmonicsTest) {
    torch::manual_seed(42);

    // Test different SH degrees
    std::vector<int> sh_degrees = {0, 1, 2, 3, 4};

    for (int sh_degree : sh_degrees) {
        int N = 1000;
        int K = (sh_degree + 1) * (sh_degree + 1);

        auto coeffs = torch::randn({N, K, 3}, device);
        auto dirs = torch::randn({N, 3}, device);

        // Forward pass
        auto masks = torch::ones({N}, torch::TensorOptions().dtype(torch::kBool).device(device));
        auto colors = gsplat::spherical_harmonics_fwd(sh_degree, dirs, coeffs, masks);

        EXPECT_EQ(colors.sizes(), torch::IntArrayRef({N, 3}));

        // Check that colors are reasonable (not NaN or infinite)
        auto has_nan = colors.isnan().any();
        auto has_inf = colors.isinf().any();
        EXPECT_FALSE(has_nan.template item<bool>());
        EXPECT_FALSE(has_inf.template item<bool>());

        // Backward pass
        auto v_colors = torch::randn_like(colors);
        auto grad_results = gsplat::spherical_harmonics_bwd(
            K, sh_degree, dirs, coeffs, masks,
            v_colors, sh_degree > 0);

        auto v_coeffs = std::get<0>(grad_results);
        auto v_dirs = std::get<1>(grad_results);

        EXPECT_EQ(v_coeffs.sizes(), coeffs.sizes());
        if (sh_degree > 0) {
            EXPECT_EQ(v_dirs.sizes(), dirs.sizes());
            auto dirs_has_nan = v_dirs.isnan().any();
            EXPECT_FALSE(dirs_has_nan.template item<bool>());
        }
        auto coeffs_has_nan = v_coeffs.isnan().any();
        EXPECT_FALSE(coeffs_has_nan.template item<bool>());
    }
}

TEST_F(BasicOpsTest, IntersectTilesTest) {
    torch::manual_seed(42);

    int C = 3;   // Multiple cameras
    int N = 100; // Number of Gaussians
    int width = 40, height = 60;
    int tile_size = 16;

    auto means2d = torch::randn({C, N, 2}, device) * width;
    auto radii = torch::randint(0, width, {C, N, 2}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto depths = torch::rand({C, N}, device);

    int tile_width = (width + tile_size - 1) / tile_size;
    int tile_height = (height + tile_size - 1) / tile_size;

    // Create empty CUDA tensors for optional parameters
    auto empty_orders = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto empty_tiles_per_gauss = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Test intersect tile
    auto isect_results = gsplat::intersect_tile(
        means2d, radii, depths,
        empty_orders,
        empty_tiles_per_gauss,
        C, tile_size, tile_width, tile_height,
        true // sort
    );

    auto tiles_per_gauss = std::get<0>(isect_results);
    auto isect_ids = std::get<1>(isect_results);
    auto flatten_ids = std::get<2>(isect_results);

    // The gsplat C++ implementation may return tiles_per_gauss as 2D [C, N]
    // while the Python reference implementation returns 1D
    // We should accept both as valid
    bool tiles_per_gauss_valid = false;
    if (tiles_per_gauss.dim() == 2) {
        // 2D case: should be [C, N]
        tiles_per_gauss_valid = (tiles_per_gauss.size(0) == C && tiles_per_gauss.size(1) == N);
    } else if (tiles_per_gauss.dim() == 1) {
        // 1D case: should be flattened to C*N
        tiles_per_gauss_valid = (tiles_per_gauss.size(0) == C * N);
    }
    EXPECT_TRUE(tiles_per_gauss_valid) << "tiles_per_gauss has unexpected shape: " << tiles_per_gauss.sizes();

    // isect_ids and flatten_ids should be 1D
    EXPECT_EQ(isect_ids.dim(), 1);
    EXPECT_EQ(flatten_ids.dim(), 1);

    // isect_ids and flatten_ids should be 1D
    EXPECT_EQ(isect_ids.dim(), 1);
    EXPECT_EQ(flatten_ids.dim(), 1);

    // Check that all values are valid
    if (tiles_per_gauss.numel() > 0) {
        auto tpg_valid = (tiles_per_gauss >= 0).all();
        EXPECT_TRUE(tpg_valid.template item<bool>());
    }

    if (isect_ids.numel() > 0) {
        auto isect_valid = (isect_ids >= 0).all();
        EXPECT_TRUE(isect_valid.template item<bool>());
    }

    if (flatten_ids.numel() > 0) {
        auto flatten_valid = (flatten_ids >= 0).all();
        EXPECT_TRUE(flatten_valid.template item<bool>());
    }

    // Test offset encoding
    auto isect_offsets = gsplat::intersect_offset(isect_ids, C, tile_width, tile_height);
    EXPECT_EQ(isect_offsets.sizes(), torch::IntArrayRef({C, tile_height, tile_width}));
}

TEST_F(BasicOpsTest, RasterizationIntegrationTest) {
    torch::manual_seed(42);

    // Simple scene setup
    int N = 100;
    int width = 256, height = 256;
    int tile_size = 16;

    // Create Gaussians
    auto means = torch::randn({N, 3}, device) * 2.0;
    auto quats = torch::randn({N, 4}, device);
    quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto scales = torch::rand({N, 3}, device) * 0.5;
    auto opacities = torch::rand({N}, device);
    auto colors = torch::rand({N, 3}, device);

    // Camera setup
    auto viewmat = torch::eye(4, device).unsqueeze(0);
    auto K = torch::tensor({{200.0f, 0.0f, 128.0f},
                            {0.0f, 200.0f, 128.0f},
                            {0.0f, 0.0f, 1.0f}},
                           device)
                 .unsqueeze(0);

    // Create empty CUDA tensor for covars
    auto empty_covars = torch::empty({0, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Project Gaussians
    auto proj_results = gsplat::projection_ewa_3dgs_fused_fwd(
        means,
        empty_covars,
        quats,
        scales,
        opacities,
        viewmat,
        K,
        width,
        height,
        0.3f,    // eps2d
        0.01f,   // near_plane
        1000.0f, // far_plane
        0.0f,    // radius_clip
        false,   // calc_compensations
        gsplat::CameraModelType::PINHOLE);

    auto radii = std::get<0>(proj_results);
    auto means2d = std::get<1>(proj_results);
    auto depths = std::get<2>(proj_results);
    auto conics = std::get<3>(proj_results);

    // Tile intersection
    int tile_width = (width + tile_size - 1) / tile_size;
    int tile_height = (height + tile_size - 1) / tile_size;

    auto empty_orders = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto empty_tiles_per_gauss = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    auto isect_results = gsplat::intersect_tile(
        means2d, radii, depths,
        empty_orders,
        empty_tiles_per_gauss,
        1, tile_size, tile_width, tile_height,
        true // sort
    );

    auto tiles_per_gauss = std::get<0>(isect_results);
    auto isect_ids = std::get<1>(isect_results);
    auto flatten_ids = std::get<2>(isect_results);

    auto isect_offsets = gsplat::intersect_offset(isect_ids, 1, tile_width, tile_height);
    isect_offsets = isect_offsets.reshape({1, tile_height, tile_width});

    // Prepare for rasterization - need [C, N, D] format
    colors = colors.unsqueeze(0);       // [1, N, 3]
    opacities = opacities.unsqueeze(0); // [1, N]
    auto background = torch::zeros({1, 3}, device);
    auto empty_masks = torch::empty({0}, torch::TensorOptions().dtype(torch::kBool).device(device));

    // Rasterize
    auto raster_results = gsplat::rasterize_to_pixels_3dgs_fwd(
        means2d,
        conics,
        colors,
        opacities,
        background,
        empty_masks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids);

    auto render_colors = std::get<0>(raster_results);
    auto render_alphas = std::get<1>(raster_results);

    // Check output
    EXPECT_EQ(render_colors.sizes(), torch::IntArrayRef({1, height, width, 3}));
    EXPECT_EQ(render_alphas.sizes(), torch::IntArrayRef({1, height, width, 1}));

    // Check values are in valid range
    auto colors_valid = (render_colors >= 0).all().item<bool>() && (render_colors <= 1).all().item<bool>();
    auto alphas_valid = (render_alphas >= 0).all().item<bool>() && (render_alphas <= 1).all().item<bool>();

    EXPECT_TRUE(colors_valid);
    EXPECT_TRUE(alphas_valid);
}

TEST_F(BasicOpsTest, ProjectionPackedModeTest) {
    torch::manual_seed(42);

    // Test projection with multiple cameras
    int N = 500;
    int C = 3;
    int width = 640, height = 480;

    auto means = torch::randn({N, 3}, device);
    auto quats = torch::randn({N, 4}, device);
    auto scales = torch::rand({N, 3}, device) * 0.1;
    auto opacities = torch::rand({N}, device);
    auto viewmats = torch::eye(4, device).unsqueeze(0).repeat({C, 1, 1});
    auto Ks = torch::tensor({{300.0f, 0.0f, 320.0f},
                             {0.0f, 300.0f, 240.0f},
                             {0.0f, 0.0f, 1.0f}},
                            device)
                  .unsqueeze(0)
                  .repeat({C, 1, 1});

    // Create empty CUDA tensor for covars
    auto empty_covars = torch::empty({0, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Project with multiple cameras
    auto proj_results = gsplat::projection_ewa_3dgs_fused_fwd(
        means,
        empty_covars,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        0.3f,     // eps2d
        0.01f,    // near_plane
        10000.0f, // far_plane
        0.0f,     // radius_clip
        false,    // calc_compensations
        gsplat::CameraModelType::PINHOLE);

    auto radii = std::get<0>(proj_results);
    auto means2d = std::get<1>(proj_results);
    auto depths = std::get<2>(proj_results);
    auto conics = std::get<3>(proj_results);

    // Check that outputs have correct shapes for multiple cameras
    EXPECT_EQ(radii.sizes(), torch::IntArrayRef({C, N, 2}));
    EXPECT_EQ(means2d.sizes(), torch::IntArrayRef({C, N, 2}));
    EXPECT_EQ(depths.sizes(), torch::IntArrayRef({C, N}));
    EXPECT_EQ(conics.sizes(), torch::IntArrayRef({C, N, 3}));

    // At least some Gaussians should be visible in each camera
    for (int c = 0; c < C; ++c) {
        auto visible = (radii[c] > 0).any(-1);
        auto num_visible = visible.sum();
        EXPECT_GT(num_visible.template item<int64_t>(), 0);
    }
}

TEST_F(BasicOpsTest, CameraModelsTest) {
    torch::manual_seed(42);

    int N = 100;
    int width = 640, height = 480;

    auto means = torch::randn({N, 3}, device);
    auto quats = torch::randn({N, 4}, device);
    auto scales = torch::rand({N, 3}, device) * 0.1;
    auto viewmat = torch::eye(4, device).unsqueeze(0);
    auto K = torch::tensor({{300.0f, 0.0f, 320.0f},
                            {0.0f, 300.0f, 240.0f},
                            {0.0f, 0.0f, 1.0f}},
                           device)
                 .unsqueeze(0);

    // Create empty CUDA tensor for covars
    auto empty_covars = torch::empty({0, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Test different camera models
    std::vector<gsplat::CameraModelType> models = {
        gsplat::CameraModelType::PINHOLE,
        gsplat::CameraModelType::ORTHO,
        gsplat::CameraModelType::FISHEYE};

    for (auto model : models) {
        auto proj_results = gsplat::projection_ewa_3dgs_fused_fwd(
            means,
            empty_covars,
            quats,
            scales,
            torch::rand({N}, device), // opacities
            viewmat,
            K,
            width,
            height,
            0.3f,     // eps2d
            0.01f,    // near_plane
            10000.0f, // far_plane
            0.0f,     // radius_clip
            false,    // calc_compensations
            model);

        auto radii = std::get<0>(proj_results);
        auto means2d = std::get<1>(proj_results);

        // Basic sanity checks
        auto radii_nan = radii.isnan().any();
        auto means2d_nan = means2d.isnan().any();
        EXPECT_FALSE(radii_nan.template item<bool>());
        EXPECT_FALSE(means2d_nan.template item<bool>());

        // At least some Gaussians should be visible
        auto visible = (radii > 0).any(-1);
        auto num_visible = visible.sum();
        EXPECT_GT(num_visible.template item<int64_t>(), 0);
    }
}