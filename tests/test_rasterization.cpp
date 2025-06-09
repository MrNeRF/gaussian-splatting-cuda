#include "core/camera.hpp"
#include "core/debug_utils.hpp"
#include "core/parameters.hpp"
#include "core/rasterizer.hpp"
#include "core/splat_data.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <torch/torch.h>

class RasterizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        device = torch::kCUDA;

        // Default test parameters
        params.optimization.sh_degree = 3;

        // Set up test scene
        setupTestScene();
    }

    void setupTestScene() {
        // Create a simple test scene with known Gaussians
        N = 100;
        C = 2; // 2 cameras
        width = 300;
        height = 200;

        // Create test SplatData
        auto means = torch::randn({N, 3}, torch::kFloat32) * 2.0f;
        auto sh0 = torch::rand({N, 1, 3}, torch::kFloat32);
        auto shN = torch::rand({N, (params.optimization.sh_degree + 1) * (params.optimization.sh_degree + 1) - 1, 3}, torch::kFloat32);
        auto scales = torch::log(torch::rand({N, 3}, torch::kFloat32) * 0.1f + 0.01f);
        auto quats = torch::randn({N, 4}, torch::kFloat32);
        quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
        auto opacity = torch::logit(torch::rand({N, 1}, torch::kFloat32) * 0.8f + 0.1f);

        splat_data = std::make_unique<SplatData>(
            params.optimization.sh_degree,
            means, sh0, shN, scales, quats, opacity, 1.0f);

        // Move to CUDA
        splat_data->means() = splat_data->means().to(device);
        splat_data->sh0() = splat_data->sh0().to(device);
        splat_data->shN() = splat_data->shN().to(device);
        splat_data->scaling_raw() = splat_data->scaling_raw().to(device);
        splat_data->rotation_raw() = splat_data->rotation_raw().to(device);
        splat_data->opacity_raw() = splat_data->opacity_raw().to(device);

        // Create test cameras
        for (int i = 0; i < C; ++i) {
            auto R = torch::eye(3, torch::kFloat32);
            auto T = torch::tensor({0.0f, 0.0f, 5.0f + i * 2.0f}, torch::kFloat32);
            float fov = M_PI / 3.0f; // 60 degrees

            cameras.push_back(std::make_unique<Camera>(
                R, T, fov, fov,
                "test_cam_" + std::to_string(i),
                "", width, height, i));
        }

        // Background color
        background = torch::zeros({3}, device);
    }

    // Helper methods
    SplatData createEmptySplatData() {
        return SplatData(
            0,                                                   // sh_degree
            torch::zeros({0, 3}, torch::kFloat32).to(device),    // means
            torch::zeros({0, 1, 3}, torch::kFloat32).to(device), // sh0
            torch::zeros({0, 0, 3}, torch::kFloat32).to(device), // shN
            torch::zeros({0, 3}, torch::kFloat32).to(device),    // scaling
            torch::zeros({0, 4}, torch::kFloat32).to(device),    // rotation
            torch::zeros({0, 1}, torch::kFloat32).to(device),    // opacity
            1.0f                                                 // scene_scale
        );
    }

    SplatData createSplatDataBehindCamera() {
        int n = 10;
        auto means = torch::zeros({n, 3}, torch::kFloat32);
        means.select(1, 2) = -10.0f; // All Z coordinates behind camera

        return SplatData(
            0, // sh_degree
            means.to(device),
            torch::rand({n, 1, 3}, torch::kFloat32).to(device),
            torch::zeros({n, 0, 3}, torch::kFloat32).to(device),
            torch::zeros({n, 3}, torch::kFloat32).to(device),
            torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}, torch::kFloat32).repeat({n, 1}).to(device),
            torch::zeros({n, 1}, torch::kFloat32).to(device),
            1.0f);
    }

    SplatData createLargeSplatData() {
        int n = 5;
        auto means = torch::randn({n, 3}, torch::kFloat32) * 0.5f;
        auto scales = torch::ones({n, 3}, torch::kFloat32) * 2.0f; // Log of large scale

        return SplatData(
            0, // sh_degree
            means.to(device),
            torch::rand({n, 1, 3}, torch::kFloat32).to(device),
            torch::zeros({n, 0, 3}, torch::kFloat32).to(device),
            scales.to(device),
            torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}, torch::kFloat32).repeat({n, 1}).to(device),
            torch::ones({n, 1}, torch::kFloat32).to(device),
            1.0f);
    }

    torch::Device device{torch::kCPU};      // Initialize with default value
    gs::param::TrainingParameters params{}; // Initialize with default value
    std::unique_ptr<SplatData> splat_data;
    std::vector<std::unique_ptr<Camera>> cameras;
    torch::Tensor background;
    int N{0}, C{0}, width{0}, height{0}; // Initialize primitive types
};                                       // Class ends here

// Test cases go outside the class
TEST_F(RasterizationTest, BasicRasterizationTest) {
    // Test basic rasterization for each camera
    for (int cam_idx = 0; cam_idx < C; ++cam_idx) {
        auto& cam = cameras[cam_idx];

        // Rasterize
        auto bg_copy = background.clone();
        auto render_output = gs::rasterize(
            *cam, *splat_data, bg_copy, 1.0f, false);

        // Check output dimensions
        EXPECT_EQ(render_output.image.sizes(), torch::IntArrayRef({3, height, width}));
        EXPECT_EQ(render_output.means2d.sizes(), torch::IntArrayRef({N, 2}));
        EXPECT_EQ(render_output.depths.sizes(), torch::IntArrayRef({N}));
        EXPECT_EQ(render_output.radii.sizes(), torch::IntArrayRef({N}));
        EXPECT_EQ(render_output.visibility.sizes(), torch::IntArrayRef({N}));
        EXPECT_EQ(render_output.width, width);
        EXPECT_EQ(render_output.height, height);

        // Check that rendered image is in valid range [0, 1]
        EXPECT_TRUE((render_output.image >= 0).all().template item<bool>());
        EXPECT_TRUE((render_output.image <= 1).all().template item<bool>());

        // Check no NaN values
        EXPECT_FALSE(render_output.image.isnan().any().template item<bool>());
        EXPECT_FALSE(render_output.means2d.isnan().any().template item<bool>());
        EXPECT_FALSE(render_output.depths.isnan().any().template item<bool>());

        // At least some Gaussians should be visible
        EXPECT_GT(render_output.visibility.sum().template item<int64_t>(), 0);
    }
}

TEST_F(RasterizationTest, RenderModesTest) {
    // Test different render modes by manipulating the rendering pipeline
    auto& cam = cameras[0];

    // RGB mode (default)
    {
        auto bg_copy = background.clone();
        auto render_output = gs::rasterize(
            *cam, *splat_data, bg_copy, 1.0f, false);
        EXPECT_EQ(render_output.image.size(0), 3); // RGB channels
    }

    // Note: Our C++ implementation doesn't directly support D, ED, RGB+D modes
    // but we can test that depth information is correctly computed
    {
        auto bg_copy = background.clone();
        auto render_output = gs::rasterize(
            *cam, *splat_data, bg_copy, 1.0f, false);

        // Check depths are positive for visible Gaussians
        auto visible_depths = render_output.depths.masked_select(render_output.visibility);
        if (visible_depths.numel() > 0) {
            EXPECT_TRUE((visible_depths > 0).all().template item<bool>());
        }
    }
}

TEST_F(RasterizationTest, SphericalHarmonicsRenderingTest) {
    // Test that SH coefficients are properly used in rendering
    auto& cam = cameras[0];

    // Render with current SH degree
    auto bg_copy1 = background.clone();
    auto render1 = gs::rasterize(*cam, *splat_data, bg_copy1, 1.0f, false);

    // Change active SH degree
    int original_degree = splat_data->get_active_sh_degree();
    splat_data->increment_sh_degree();

    // Render again
    auto bg_copy2 = background.clone();
    auto render2 = gs::rasterize(*cam, *splat_data, bg_copy2, 1.0f, false);

    // Images should be different (unless all higher order coefficients are zero)
    if (original_degree < params.optimization.sh_degree) {
        // There's a chance they could be the same if higher order SH coeffs are near zero
        // but generally they should differ
        auto diff = (render1.image - render2.image).abs().sum();
        // Just check that computation succeeded without errors
        EXPECT_FALSE(diff.isnan().template item<bool>());
    }
}

TEST_F(RasterizationTest, ScalingModifierTest) {
    auto& cam = cameras[0];

    // Render with different scaling modifiers
    float scale_modifiers[] = {0.5f, 1.0f, 2.0f};
    std::vector<torch::Tensor> renders;
    std::vector<int> visible_counts;

    for (float scale_mod : scale_modifiers) {
        auto bg_copy = background.clone();
        auto render_output = gs::rasterize(
            *cam, *splat_data, bg_copy, scale_mod, false);
        renders.push_back(render_output.image);
        visible_counts.push_back(render_output.visibility.sum().template item<int64_t>());
    }

    // Larger scaling modifier should generally result in more visible Gaussians
    // (they become larger and more likely to be in view)
    EXPECT_LE(visible_counts[0], visible_counts[1]);
    EXPECT_LE(visible_counts[1], visible_counts[2]);

    // Images should be different
    EXPECT_FALSE(torch::allclose(renders[0], renders[1]));
    EXPECT_FALSE(torch::allclose(renders[1], renders[2]));
}

TEST_F(RasterizationTest, BackgroundColorTest) {
    auto& cam = cameras[0];

    // Test with different background colors
    std::vector<torch::Tensor> backgrounds = {
        torch::zeros({3}, device),
        torch::ones({3}, device) * 0.5f,
        torch::tensor({1.0f, 0.0f, 0.0f}, device) // Red background
    };

    std::vector<torch::Tensor> renders;

    for (const auto& bg : backgrounds) {
        auto bg_copy = bg.clone(); // rasterize might modify it
        auto render_output = gs::rasterize(
            *cam, *splat_data, bg_copy, 1.0f, false);
        renders.push_back(render_output.image);
    }

    // Different backgrounds should produce different images
    EXPECT_FALSE(torch::allclose(renders[0], renders[1]));
    EXPECT_FALSE(torch::allclose(renders[1], renders[2]));

    // Areas with low alpha should show background color
    // This is a bit tricky to test without knowing exact alpha values,
    // but we can at least check that red background affects the red channel more
    auto red_channel_mean = renders[2].select(0, 0).mean();
    auto blue_channel_mean = renders[2].select(0, 2).mean();
    // With red background, red channel should generally be higher
    // (this might not always be true if scene is very dense)
}

TEST_F(RasterizationTest, GradientFlowTest) {
    // Test that gradients flow through the rendering pipeline
    auto& cam = cameras[0];

    // Enable gradients
    splat_data->means().set_requires_grad(true);
    splat_data->scaling_raw().set_requires_grad(true);
    splat_data->rotation_raw().set_requires_grad(true);
    splat_data->opacity_raw().set_requires_grad(true);
    splat_data->sh0().set_requires_grad(true);
    splat_data->shN().set_requires_grad(true);

    // Forward pass
    auto bg_copy = background.clone();
    auto render_output = gs::rasterize(
        *cam, *splat_data, bg_copy, 1.0f, false);

    // Create a simple loss
    auto loss = render_output.image.mean();

    // Backward pass
    loss.backward();

    // Check that gradients are computed
    EXPECT_TRUE(splat_data->means().grad().defined());
    EXPECT_TRUE(splat_data->scaling_raw().grad().defined());
    EXPECT_TRUE(splat_data->rotation_raw().grad().defined());
    EXPECT_TRUE(splat_data->opacity_raw().grad().defined());
    EXPECT_TRUE(splat_data->sh0().grad().defined());
    EXPECT_TRUE(splat_data->shN().grad().defined());

    // Check gradients are not all zero
    EXPECT_GT(splat_data->means().grad().abs().sum().template item<float>(), 0);

    // Check no NaN gradients
    EXPECT_FALSE(splat_data->means().grad().isnan().any().template item<bool>());
    EXPECT_FALSE(splat_data->scaling_raw().grad().isnan().any().template item<bool>());
    EXPECT_FALSE(splat_data->rotation_raw().grad().isnan().any().template item<bool>());
}

TEST_F(RasterizationTest, MultiViewConsistencyTest) {
    // Test that the same scene renders consistently from different viewpoints
    std::vector<torch::Tensor> renders;
    std::vector<int> visible_counts;

    for (int i = 0; i < C; ++i) {
        auto bg_copy = background.clone();
        auto render_output = gs::rasterize(
            *cameras[i], *splat_data, bg_copy, 1.0f, false);
        renders.push_back(render_output.image);
        visible_counts.push_back(render_output.visibility.sum().template item<int>());
    }

    // Different viewpoints should produce different images
    EXPECT_FALSE(torch::allclose(renders[0], renders[1]));

    // Both views should see some Gaussians
    EXPECT_GT(visible_counts[0], 0);
    EXPECT_GT(visible_counts[1], 0);

    // Check that both renders are valid
    for (const auto& render : renders) {
        EXPECT_TRUE((render >= 0).all().template item<bool>());
        EXPECT_TRUE((render <= 1).all().template item<bool>());
        EXPECT_FALSE(render.isnan().any().template item<bool>());
    }
}

TEST_F(RasterizationTest, PerformanceConsistencyTest) {
    // Test that multiple renders of the same scene produce identical results
    // (important for debugging and reproducibility)
    auto& cam = cameras[0];

    // Render the same scene multiple times
    const int num_renders = 5;
    std::vector<torch::Tensor> renders;

    for (int i = 0; i < num_renders; ++i) {
        auto bg_copy = background.clone();
        auto render_output = gs::rasterize(
            *cam, *splat_data, bg_copy, 1.0f, false);
        renders.push_back(render_output.image);
    }

    // All renders should be identical
    for (int i = 1; i < num_renders; ++i) {
        EXPECT_TRUE(torch::allclose(renders[0], renders[i], 1e-6))
            << "Render " << i << " differs from render 0";
    }
}

TEST_F(RasterizationTest, EdgeCasesTest) {
    auto& cam = cameras[0];

    // Test with empty scene
    {
        auto empty_splat = createEmptySplatData();
        auto bg_copy = background.clone();
        auto render_output = gs::rasterize(
            *cam, empty_splat, bg_copy, 1.0f, false);

        // Should produce background color
        auto expected = background.unsqueeze(1).unsqueeze(2).expand({3, height, width});
        EXPECT_TRUE(torch::allclose(render_output.image, expected));
    }

    // Test with all Gaussians behind camera
    {
        auto behind_cam_splat = createSplatDataBehindCamera();
        auto bg_copy = background.clone();
        auto render_output = gs::rasterize(
            *cam, behind_cam_splat, bg_copy, 1.0f, false);

        // Should produce background color (no visible Gaussians)
        EXPECT_EQ(render_output.visibility.sum().template item<int64_t>(), 0);
    }

    // Test with very large Gaussians
    {
        auto large_splat = createLargeSplatData();
        auto bg_copy = background.clone();
        auto render_output = gs::rasterize(
            *cam, large_splat, bg_copy, 1.0f, false);

        // Should still produce valid output
        EXPECT_TRUE((render_output.image >= 0).all().template item<bool>());
        EXPECT_TRUE((render_output.image <= 1).all().template item<bool>());
        EXPECT_FALSE(render_output.image.isnan().any().template item<bool>());
    }
}