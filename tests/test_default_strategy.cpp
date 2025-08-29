#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "rasterization/rasterizer.hpp"
#include "strategies/default_strategy.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <torch/torch.h>

using namespace gs;

class DefaultStrategyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up CUDA device
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        device = torch::kCUDA;

        // Create dummy parameters for testing
        params.optimization.iterations = 30000;
        params.optimization.means_lr = 1e-3f;
        params.optimization.shs_lr = 1e-3f;
        params.optimization.opacity_lr = 0.05f;
        params.optimization.scaling_lr = 5e-3f;
        params.optimization.rotation_lr = 1e-3f;
        params.optimization.max_cap = 10000;
        params.optimization.min_opacity = 0.005f;
        params.optimization.start_refine = 500;
        params.optimization.stop_refine = 25000;
        params.optimization.refine_every = 100;
        params.optimization.sh_degree = 3;

        // Set up test camera
        setupTestCamera();
    }

    void setupTestCamera() {
        // Create a test camera
        auto R = torch::eye(3, torch::kFloat32);
        auto T = torch::tensor({0.0f, 0.0f, 5.0f}, torch::kFloat32);
        float fov = M_PI / 3.0f; // 60 degrees

        int width = 256;
        int height = 256;

        test_camera = std::make_unique<Camera>(
            R, T, fov2focal(fov, width),
            fov2focal(fov, height),
            0.5 * width,
            0.5 * height,
            torch::empty({0}, torch::kFloat32),
            torch::empty({0}, torch::kFloat32),
            gsplat::CameraModelType::PINHOLE,
            "test_camera",
            "", width, height, 0);

        // Background color
        background = torch::zeros({3}, device);
    }

    // Helper to create a dummy SplatData for testing
    SplatData createTestSplatData(int N) {
        // Create tensors without gradients for testing
        torch::NoGradGuard no_grad;

        auto means = torch::randn({N, 3}, torch::kFloat32);
        auto sh0 = torch::randn({N, 1, 3}, torch::kFloat32);
        auto shN = torch::randn({N, (params.optimization.sh_degree + 1) * (params.optimization.sh_degree + 1) - 1, 3}, torch::kFloat32);
        auto scaling = torch::randn({N, 3}, torch::kFloat32);
        auto rotation = torch::randn({N, 4}, torch::kFloat32);
        auto opacity = torch::randn({N, 1}, torch::kFloat32);

        return SplatData(params.optimization.sh_degree, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    // Helper to perform actual rendering
    gs::RenderOutput performRendering(DefaultStrategy& strategy) {
        auto bg_copy = background.clone();
        return gs::rasterize(*test_camera, strategy.get_model(), bg_copy, 1.0f, false);
    }

    torch::Device device{torch::kCPU};
    gs::param::TrainingParameters params{};
    std::unique_ptr<Camera> test_camera;
    torch::Tensor background;
};

TEST_F(DefaultStrategyTest, FullPipelineIntegrationTest) {
    // Create test data
    auto splat_data = createTestSplatData(100);
    auto strategy = std::make_unique<DefaultStrategy>(std::move(splat_data));
    strategy->initialize(params.optimization);

    // Perform actual rendering
    auto render_output = performRendering(*strategy);

    // Check render output is valid
    EXPECT_EQ(render_output.image.sizes(), torch::IntArrayRef({3, 256, 256}));
    EXPECT_FALSE(render_output.image.isnan().any().item<bool>());

    // Compute loss and backward
    auto loss = render_output.image.mean();
    loss.backward();

    // Check that gradients are computed for all parameters
    EXPECT_TRUE(strategy->get_model().means().grad().defined());
    EXPECT_TRUE(strategy->get_model().opacity_raw().grad().defined());
    EXPECT_TRUE(strategy->get_model().scaling_raw().grad().defined());
    EXPECT_TRUE(strategy->get_model().rotation_raw().grad().defined());
    EXPECT_TRUE(strategy->get_model().sh0().grad().defined());
    EXPECT_TRUE(strategy->get_model().shN().grad().defined());

    // Run default strategy step
    strategy->post_backward(600, render_output);
    strategy->step(600);

    // Verify step was taken
    EXPECT_FALSE(strategy->get_model().means().grad().defined()) << "Gradients should be zeroed after step";
}

TEST_F(DefaultStrategyTest, InitializationTest) {
    // Create test data
    auto splat_data = createTestSplatData(100);

    // Create MCMC strategy
    auto strategy = std::make_unique<DefaultStrategy>(std::move(splat_data));

    // Initialize
    ASSERT_NO_THROW(strategy->initialize(params.optimization));

    // Check that parameters are on CUDA and require grad
    EXPECT_TRUE(strategy->get_model().means().is_cuda());
    EXPECT_TRUE(strategy->get_model().means().requires_grad());
    EXPECT_TRUE(strategy->get_model().scaling_raw().is_cuda());
    EXPECT_TRUE(strategy->get_model().scaling_raw().requires_grad());
    EXPECT_TRUE(strategy->get_model().rotation_raw().is_cuda());
    EXPECT_TRUE(strategy->get_model().rotation_raw().requires_grad());
}

TEST_F(DefaultStrategyTest, SHDegreeIncrementWithRenderingTest) {
    // Create test data
    auto splat_data = createTestSplatData(50);
    auto strategy = std::make_unique<DefaultStrategy>(std::move(splat_data));

    // Set parameters to avoid refinement at iteration 1000
    params.optimization.start_refine = 1001; // Start after iteration 1000
    params.optimization.stop_refine = 2000;  // Stop densification later
    params.optimization.refine_every = 100;
    strategy->initialize(params.optimization);

    // Perform initial render to establish optimizer states
    auto render_output = performRendering(*strategy);
    auto loss = render_output.image.mean();
    loss.backward();
    strategy->step(1);

    // Get initial SH degree
    int initial_degree = strategy->get_model().get_active_sh_degree();

    // Render again and call post_backward at iteration 1000
    render_output = performRendering(*strategy);
    loss = render_output.image.mean();
    loss.backward();

    strategy->post_backward(1000, render_output);

    // Check SH degree increased
    int new_degree = strategy->get_model().get_active_sh_degree();
    EXPECT_EQ(new_degree, initial_degree + 1);
}

TEST_F(DefaultStrategyTest, GradientFlowTest) {
    // Test that gradients flow correctly through the full pipeline
    auto splat_data = createTestSplatData(50);
    auto strategy = std::make_unique<DefaultStrategy>(std::move(splat_data));
    strategy->initialize(params.optimization);

    // Store initial values
    auto means_before = strategy->get_model().means().clone();

    // Render and compute loss
    auto render_output = performRendering(*strategy);
    auto loss = render_output.image.sum(); // Use sum for stronger gradients
    loss.backward();

    // Check gradients exist and are non-zero
    auto means2d_grad = render_output.means2d.grad();
    EXPECT_TRUE(means2d_grad.defined());
    auto means2d_grad_norm = means2d_grad.norm();
    EXPECT_GT(means2d_grad_norm.item<float>(), 0) << "Gradients should be non-zero";

    auto means_grad = strategy->get_model().means().grad();
    EXPECT_TRUE(means_grad.defined());
    auto means_grad_norm = means_grad.norm();
    EXPECT_GT(means_grad_norm.item<float>(), 0) << "Gradients should be non-zero";

    // Take optimizer step
    strategy->step(1);

    // Check that parameters changed
    auto means_after = strategy->get_model().means();
    auto diff = (means_after - means_before).abs().sum();
    EXPECT_GT(diff.item<float>(), 0) << "Parameters should change after optimizer step";
}

TEST_F(DefaultStrategyTest, ResetOpacityWithRenderingTest) {
    // Test resetting opacity in the context of actual rendering
    auto splat_data = createTestSplatData(20);
    auto strategy = std::make_unique<DefaultStrategy>(std::move(splat_data));

    // Set parameters to avoid refinement at iteration 3000
    params.optimization.start_refine = 3001;
    strategy->initialize(params.optimization);

    // Perform initial render and step to initialize optimizer
    auto render_output = performRendering(*strategy);
    auto loss = render_output.image.mean();
    loss.backward();
    strategy->step(1);

    // Store initial parameters
    auto means_before = strategy->get_model().means().clone();
    auto rotation_before = strategy->get_model().get_rotation().clone();
    auto scaling_before = strategy->get_model().get_scaling().clone();
    auto shs_before = strategy->get_model().get_shs().clone();

    // Render again
    render_output = performRendering(*strategy);
    loss = render_output.image.mean();
    loss.backward();

    // Call post_backward at iteration 3000 (no growing or pruning, just resetting opacity)
    strategy->post_backward(3000, render_output);

    // Opacities should be reset
    const auto threshold = torch::logit(torch::tensor(2.0f * 0.005f));
    auto opacities = strategy->get_model().opacity_raw();
    auto max_opacity_value = opacities.max();
    EXPECT_LE(max_opacity_value.item<float>(), threshold.item<float>()) << "Opacities should be reset to low values";

    // Other parameters should be the same
    auto means_after = strategy->get_model().means();
    auto rotation_after = strategy->get_model().get_rotation();
    auto scaling_after = strategy->get_model().get_scaling();
    auto shs_after = strategy->get_model().get_shs();
    EXPECT_TRUE((means_after == means_before).all().item<bool>());
    EXPECT_TRUE((rotation_after == rotation_before).all().item<bool>());
    EXPECT_TRUE((scaling_after == scaling_before).all().item<bool>());
    EXPECT_TRUE((shs_after == shs_before).all().item<bool>());
}

TEST_F(DefaultStrategyTest, RefinementWithActualRenderingTest) {
    // Test refinement (growing and pruning) with actual rendering
    auto splat_data = createTestSplatData(30);
    auto strategy = std::make_unique<DefaultStrategy>(std::move(splat_data));

    // Set up for refinement
    params.optimization.start_refine = 500;
    params.optimization.stop_refine = 1000;
    params.optimization.refine_every = 100;
    strategy->initialize(params.optimization);

    // Initialize optimizer
    auto render_output = performRendering(*strategy);
    auto loss = render_output.image.mean();
    loss.backward();
    strategy->step(1);

    int initial_size = strategy->get_model().size();

    // Run refinement step
    render_output = performRendering(*strategy);
    loss = render_output.image.mean();
    loss.backward();

    strategy->post_backward(600, render_output);

    // Size might have changed due to refinement
    int new_size = strategy->get_model().size();
    std::cout << "Size changed from " << initial_size << " to " << new_size << std::endl;
}

TEST_F(DefaultStrategyTest, DuplicationMechanicsTest) {
    // Detailed test of duplication with actual rendering
    auto splat_data = createTestSplatData(100);
    auto strategy = std::make_unique<DefaultStrategy>(std::move(splat_data));

    params.optimization.start_refine = 500;
    params.optimization.stop_refine = 1000;

    // Set grow_scale3d to a very high value to avoid splitting
    params.optimization.grow_scale3d = 100;
    params.optimization.grad_threshold = 0;
    strategy->initialize(params.optimization);

    // Initialize optimizer
    auto render_output = performRendering(*strategy);
    auto loss = render_output.image.mean();
    loss.backward();
    strategy->step(1);

    // Store initial parameters
    int initial_size = strategy->get_model().size();
    auto means_before = strategy->get_model().means().clone();
    auto opacity_before = strategy->get_model().get_opacity().clone();
    auto rotation_before = strategy->get_model().get_rotation().clone();
    auto scaling_before = strategy->get_model().get_scaling().clone();
    auto shs_before = strategy->get_model().get_shs().clone();

    // Trigger duplication through rendering
    render_output = performRendering(*strategy);
    loss = render_output.image.mean();
    loss.backward();

    strategy->post_backward(600, render_output);

    // Check if the number of Gaussians increased
    int new_size = strategy->get_model().size();
    EXPECT_GE(new_size, initial_size) << "Number of Gaussians should increase after duplication";

    // Check that the initial parameters did not change
    auto means_after = strategy->get_model().means();
    auto opacity_after = strategy->get_model().get_opacity();
    auto rotation_after = strategy->get_model().get_rotation();
    auto scaling_after = strategy->get_model().get_scaling();
    auto shs_after = strategy->get_model().get_shs();

    EXPECT_TRUE((means_after.slice(0, 0, initial_size) == means_before).all().item<bool>());
    EXPECT_TRUE((opacity_after.slice(0, 0, initial_size) == opacity_before).all().item<bool>());
    EXPECT_TRUE((rotation_after.slice(0, 0, initial_size) == rotation_before).all().item<bool>());
    EXPECT_TRUE((scaling_after.slice(0, 0, initial_size) == scaling_before).all().item<bool>());
    EXPECT_TRUE((shs_after.slice(0, 0, initial_size) == shs_before).all().item<bool>());
}

TEST_F(DefaultStrategyTest, SplittingMechanicsTest) {
    // Detailed test of splitting with actual rendering
    auto splat_data = createTestSplatData(100);
    auto strategy = std::make_unique<DefaultStrategy>(std::move(splat_data));

    params.optimization.start_refine = 500;
    params.optimization.stop_refine = 1000;

    // Set grow_scale3d to a very low value to avoid duplication
    params.optimization.grow_scale3d = 0;
    params.optimization.grad_threshold = 0;
    strategy->initialize(params.optimization);

    // Initialize optimizer
    auto render_output = performRendering(*strategy);
    auto loss = render_output.image.mean();
    loss.backward();
    strategy->step(1);

    // Store initial parameters
    int initial_size = strategy->get_model().size();
    auto means_before = strategy->get_model().means().clone();
    auto opacity_before = strategy->get_model().get_opacity().clone();
    auto rotation_before = strategy->get_model().get_rotation().clone();
    auto scaling_before = strategy->get_model().get_scaling().clone();
    auto shs_before = strategy->get_model().get_shs().clone();

    // Trigger duplication through rendering
    render_output = performRendering(*strategy);
    loss = render_output.image.mean();
    loss.backward();

    strategy->post_backward(600, render_output);

    // Check if the number of Gaussians increased
    int new_size = strategy->get_model().size();
    EXPECT_GE(new_size, initial_size) << "Number of Gaussians should increase after splitting";
}

TEST_F(DefaultStrategyTest, PruningMechanicsTest) {
    // Detailed test of pruning with actual rendering
    auto splat_data = createTestSplatData(100);
    auto strategy = std::make_unique<DefaultStrategy>(std::move(splat_data));

    params.optimization.start_refine = 500;
    params.optimization.stop_refine = 1000;

    // Set grad_threshold to a very high value to avoid grow_gs()
    params.optimization.grad_threshold = 1000;
    strategy->initialize(params.optimization);

    // Initialize optimizer
    auto render_output = performRendering(*strategy);
    auto loss = render_output.image.mean();
    loss.backward();
    strategy->step(1);

    // Manually set some opacities to be very low
    {
        torch::NoGradGuard no_grad;
        auto opacity_raw = strategy->get_model().opacity_raw();
        // Set first 20 Gaussians to have very low opacity
        opacity_raw.slice(0, 0, 20).fill_(torch::logit(torch::tensor(0.001f)));
    }

    int initial_size = strategy->get_model().size();

    // Trigger duplication through rendering
    render_output = performRendering(*strategy);
    loss = render_output.image.mean();
    loss.backward();

    strategy->post_backward(600, render_output);

    // Check if the number of Gaussians increased
    int new_size = strategy->get_model().size();
    EXPECT_LE(new_size, initial_size) << "Number of Gaussians should decrease after pruning";

    // Check that all opacities are now above threshold
    auto opacities = strategy->get_model().get_opacity();
    auto min_opacity_value = opacities.min();
    EXPECT_GE(min_opacity_value.item<float>(), params.optimization.prune_opacity)
        << "All opacities should be above threshold after relocation";
}

TEST_F(DefaultStrategyTest, ConsistentRenderingAfterOperationsTest) {
    // Test that rendering remains consistent after MCMC operations
    auto splat_data = createTestSplatData(50);
    auto strategy = std::make_unique<DefaultStrategy>(std::move(splat_data));
    strategy->initialize(params.optimization);

    // Initial render
    auto render1 = performRendering(*strategy);

    // Run through several iterations
    for (int iter = 1; iter <= 5; ++iter) {
        auto render_output = performRendering(*strategy);
        auto loss = render_output.image.mean();
        loss.backward();

        strategy->post_backward(500 + iter * 100, render_output);
        strategy->step(500 + iter * 100);
    }

    // Final render
    auto render2 = performRendering(*strategy);

    // Both renders should be valid
    EXPECT_FALSE(render1.image.isnan().any().item<bool>());
    EXPECT_FALSE(render2.image.isnan().any().item<bool>());
    EXPECT_TRUE((render1.image >= 0).all().item<bool>());
    EXPECT_TRUE((render1.image <= 1).all().item<bool>());
    EXPECT_TRUE((render2.image >= 0).all().item<bool>());
    EXPECT_TRUE((render2.image <= 1).all().item<bool>());

    // Images should be different due to optimization
    EXPECT_FALSE(torch::allclose(render1.image, render2.image));
}
