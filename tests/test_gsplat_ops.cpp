#include "Ops.h"
#include "core/debug_utils.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <torch/torch.h>

class GsplatOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        device = torch::kCUDA;
    }

    torch::Device device{torch::kCPU};
};

TEST_F(GsplatOpsTest, RelocationTest) {
    torch::manual_seed(42);

    // Test data matching Python test setup
    int N = 100;
    auto opacities = torch::rand({N}, device) * 0.8f + 0.1f; // [0.1, 0.9]
    auto scales = torch::rand({N, 3}, device) * 0.5f + 0.1f; // [0.1, 0.6]

    // Create ratios as in Python - must be int32!
    auto ratios = torch::randint(1, 10, {N}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Create binomial coefficients
    const int n_max = 51;
    auto binoms = torch::zeros({n_max, n_max}, torch::kFloat32);
    auto binoms_accessor = binoms.accessor<float, 2>();
    for (int n = 0; n < n_max; ++n) {
        for (int k = 0; k <= n; ++k) {
            float binom = 1.0f;
            for (int i = 0; i < k; ++i) {
                binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
            }
            binoms_accessor[n][k] = binom;
        }
    }
    binoms = binoms.to(device);

    // Test relocation function
    auto [new_opacities, new_scales] = gsplat::relocation(
        opacities,
        scales,
        ratios,
        binoms,
        n_max);

    // Basic sanity checks
    EXPECT_EQ(new_opacities.sizes(), opacities.sizes());
    EXPECT_EQ(new_scales.sizes(), scales.sizes());
    EXPECT_FALSE(new_opacities.isnan().any().item<bool>());
    EXPECT_FALSE(new_scales.isnan().any().item<bool>());

    // Values should be in reasonable ranges
    EXPECT_TRUE((new_opacities >= 0).all().item<bool>());
    EXPECT_TRUE((new_opacities <= 1).all().item<bool>());
    EXPECT_TRUE((new_scales > 0).all().item<bool>());
}

TEST_F(GsplatOpsTest, QuatScaleToCovarPreciGradientTest) {
    torch::manual_seed(42);

    int N = 100;
    auto quats = torch::randn({N, 4}, device);
    auto scales = torch::rand({N, 3}, device) * 0.1f;

    quats.set_requires_grad(true);
    scales.set_requires_grad(true);

    // Forward pass
    auto [covars, precis] = gsplat::quat_scale_to_covar_preci_fwd(
        quats,
        scales,
        true, // compute_covar
        true, // compute_preci
        false // triu
    );

    // Create gradients
    auto v_covars = torch::randn_like(covars);
    auto v_precis = torch::randn_like(precis) * 0.01f; // Small gradient for precis

    // Backward pass
    auto [v_quats, v_scales] = gsplat::quat_scale_to_covar_preci_bwd(
        quats,
        scales,
        false, // triu
        v_covars,
        v_precis);

    // Check gradients are valid
    EXPECT_TRUE(v_quats.defined());
    EXPECT_TRUE(v_scales.defined());
    EXPECT_FALSE(v_quats.isnan().any().item<bool>());
    EXPECT_FALSE(v_scales.isnan().any().item<bool>());
    EXPECT_FALSE(v_quats.isinf().any().item<bool>());
    EXPECT_FALSE(v_scales.isinf().any().item<bool>());
}

TEST_F(GsplatOpsTest, SphericalHarmonicsGradientTest) {
    torch::manual_seed(42);

    // Test with different SH degrees like Python
    std::vector<int> sh_degrees = {0, 1, 2, 3};

    for (int sh_degree : sh_degrees) {
        int N = 1000;
        int K = (sh_degree + 1) * (sh_degree + 1);

        auto coeffs = torch::randn({N, K, 3}, device);
        auto dirs = torch::randn({N, 3}, device);
        auto masks = torch::ones({N}, torch::TensorOptions().dtype(torch::kBool).device(device));

        coeffs.set_requires_grad(true);
        dirs.set_requires_grad(true);

        // Forward
        auto colors = gsplat::spherical_harmonics_fwd(sh_degree, dirs, coeffs, masks);

        // Check forward pass
        EXPECT_EQ(colors.sizes(), torch::IntArrayRef({N, 3}));
        EXPECT_FALSE(colors.isnan().any().item<bool>());
        EXPECT_FALSE(colors.isinf().any().item<bool>());

        // Backward
        auto v_colors = torch::randn_like(colors);
        auto [v_coeffs, v_dirs] = gsplat::spherical_harmonics_bwd(
            K, sh_degree, dirs, coeffs, masks,
            v_colors, sh_degree > 0 // compute_dirs_grad only for degree > 0
        );

        // Check backward pass
        EXPECT_EQ(v_coeffs.sizes(), coeffs.sizes());
        EXPECT_FALSE(v_coeffs.isnan().any().item<bool>());

        if (sh_degree > 0) {
            EXPECT_EQ(v_dirs.sizes(), dirs.sizes());
            EXPECT_FALSE(v_dirs.isnan().any().item<bool>());
        }
    }
}

TEST_F(GsplatOpsTest, ProjectionEWATest) {
    torch::manual_seed(42);

    // Setup matching Python test
    int N = 100;
    int C = 2;
    int width = 640, height = 480;

    auto means = torch::randn({N, 3}, device);
    auto quats = torch::randn({N, 4}, device);
    quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto scales = torch::rand({N, 3}, device) * 0.1f;
    auto opacities = torch::rand({N}, device);

    auto viewmats = torch::eye(4, device).unsqueeze(0).repeat({C, 1, 1});
    auto Ks = torch::tensor({{300.0f, 0.0f, 320.0f},
                             {0.0f, 300.0f, 240.0f},
                             {0.0f, 0.0f, 1.0f}},
                            device)
                  .unsqueeze(0)
                  .repeat({C, 1, 1});

    // Empty covars tensor
    auto empty_covars = torch::empty({0, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Test projection
    auto [radii, means2d, depths, conics, compensations] = gsplat::projection_ewa_3dgs_fused_fwd(
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

    // Check outputs
    EXPECT_EQ(radii.sizes(), torch::IntArrayRef({C, N, 2}));
    EXPECT_EQ(means2d.sizes(), torch::IntArrayRef({C, N, 2}));
    EXPECT_EQ(depths.sizes(), torch::IntArrayRef({C, N}));
    EXPECT_EQ(conics.sizes(), torch::IntArrayRef({C, N, 3}));

    // Check for valid values
    EXPECT_FALSE(means2d.isnan().any().item<bool>());
    EXPECT_FALSE(depths.isnan().any().item<bool>());
    EXPECT_FALSE(conics.isnan().any().item<bool>());

    // At least some Gaussians should be visible
    auto valid = (radii > 0).all(-1);
    EXPECT_GT(valid.sum().item<int64_t>(), 0);
}

TEST_F(GsplatOpsTest, RasterizationPipelineTest) {
    torch::manual_seed(42);

    // Simple scene setup matching Python
    int N = 100;
    int width = 256, height = 256;
    int tile_size = 16;

    // Create test Gaussians
    auto means = torch::randn({N, 3}, device) * 2.0f;
    auto quats = torch::randn({N, 4}, device);
    quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto scales = torch::rand({N, 3}, device) * 0.5f;
    auto opacities = torch::rand({N}, device);
    auto colors = torch::rand({N, 3}, device);

    // Camera setup
    auto viewmat = torch::eye(4, device).unsqueeze(0);
    auto K = torch::tensor({{200.0f, 0.0f, 128.0f},
                            {0.0f, 200.0f, 128.0f},
                            {0.0f, 0.0f, 1.0f}},
                           device)
                 .unsqueeze(0);

    auto empty_covars = torch::empty({0, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Project
    auto [radii, means2d, depths, conics, compensations] = gsplat::projection_ewa_3dgs_fused_fwd(
        means,
        empty_covars,
        quats,
        scales,
        opacities,
        viewmat,
        K,
        width,
        height,
        0.3f,
        0.01f,
        1000.0f,
        0.0f,
        false,
        gsplat::CameraModelType::PINHOLE);

    // Tile intersection
    int tile_width = (width + tile_size - 1) / tile_size;
    int tile_height = (height + tile_size - 1) / tile_size;

    auto empty_orders = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto empty_tiles_per_gauss = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    auto [tiles_per_gauss, isect_ids, flatten_ids] = gsplat::intersect_tile(
        means2d, radii, depths,
        empty_orders,
        empty_tiles_per_gauss,
        1, tile_size, tile_width, tile_height,
        true // sort
    );

    auto isect_offsets = gsplat::intersect_offset(isect_ids, 1, tile_width, tile_height);
    isect_offsets = isect_offsets.reshape({1, tile_height, tile_width});

    // Prepare for rasterization
    colors = colors.unsqueeze(0);
    opacities = opacities.unsqueeze(0);
    auto background = torch::zeros({1, 3}, device);
    auto empty_masks = torch::empty({0}, torch::TensorOptions().dtype(torch::kBool).device(device));

    // Rasterize
    auto [render_colors, render_alphas, last_ids] = gsplat::rasterize_to_pixels_3dgs_fwd(
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

    // Check output
    EXPECT_EQ(render_colors.sizes(), torch::IntArrayRef({1, height, width, 3}));
    EXPECT_EQ(render_alphas.sizes(), torch::IntArrayRef({1, height, width, 1}));

    // Check values are valid
    EXPECT_TRUE((render_colors >= 0).all().item<bool>());
    EXPECT_TRUE((render_colors <= 1).all().item<bool>());
    EXPECT_TRUE((render_alphas >= 0).all().item<bool>());
    EXPECT_TRUE((render_alphas <= 1).all().item<bool>());
    EXPECT_FALSE(render_colors.isnan().any().item<bool>());
    EXPECT_FALSE(render_alphas.isnan().any().item<bool>());
}
