#include "core/camera.hpp"
#include "core/mcmc.hpp"
#include "core/parameters.hpp"
#include "core/rasterizer.hpp"
#include "core/splat_data.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <torch/torch.h>

class MCMCTest : public ::testing::Test {
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

        test_camera = std::make_unique<Camera>(
            R, T, fov, fov,
            "test_camera",
            "", 256, 256, 0);

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
    gs::RenderOutput performRendering(MCMC& mcmc) {
        auto bg_copy = background.clone();
        return gs::rasterize(*test_camera, mcmc.get_model(), bg_copy, 1.0f, false);
    }

    torch::Device device{torch::kCPU};
    gs::param::TrainingParameters params{};
    std::unique_ptr<Camera> test_camera;
    torch::Tensor background;
};

TEST_F(MCMCTest, FullPipelineIntegrationTest) {
    // Create test data
    auto splat_data = createTestSplatData(100);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));
    mcmc->initialize(params.optimization);

    // Perform actual rendering
    auto render_output = performRendering(*mcmc);

    // Check render output is valid
    EXPECT_EQ(render_output.image.sizes(), torch::IntArrayRef({3, 256, 256}));
    EXPECT_FALSE(render_output.image.isnan().any().item<bool>());

    // Compute loss and backward
    auto loss = render_output.image.mean();
    loss.backward();

    // Check that gradients are computed for all parameters
    EXPECT_TRUE(mcmc->get_model().means().grad().defined());
    EXPECT_TRUE(mcmc->get_model().opacity_raw().grad().defined());
    EXPECT_TRUE(mcmc->get_model().scaling_raw().grad().defined());
    EXPECT_TRUE(mcmc->get_model().rotation_raw().grad().defined());
    EXPECT_TRUE(mcmc->get_model().sh0().grad().defined());
    EXPECT_TRUE(mcmc->get_model().shN().grad().defined());

    // Run MCMC step
    mcmc->post_backward(600, render_output);
    mcmc->step(600);

    // Verify step was taken
    EXPECT_FALSE(mcmc->get_model().means().grad().defined()) << "Gradients should be zeroed after step";
}

TEST_F(MCMCTest, InitializationTest) {
    // Create test data
    auto splat_data = createTestSplatData(100);

    // Create MCMC strategy
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));

    // Initialize
    ASSERT_NO_THROW(mcmc->initialize(params.optimization));

    // Check that parameters are on CUDA and require grad
    EXPECT_TRUE(mcmc->get_model().means().is_cuda());
    EXPECT_TRUE(mcmc->get_model().means().requires_grad());
    EXPECT_TRUE(mcmc->get_model().scaling_raw().is_cuda());
    EXPECT_TRUE(mcmc->get_model().scaling_raw().requires_grad());
    EXPECT_TRUE(mcmc->get_model().rotation_raw().is_cuda());
    EXPECT_TRUE(mcmc->get_model().rotation_raw().requires_grad());
}

TEST_F(MCMCTest, SHDegreeIncrementWithRenderingTest) {
    // Create test data
    auto splat_data = createTestSplatData(50);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));

    // Set parameters to avoid refinement at iteration 1000
    params.optimization.start_refine = 1001; // Start after iteration 1000
    params.optimization.stop_refine = 2000;  // Stop densification later
    params.optimization.refine_every = 100;
    mcmc->initialize(params.optimization);

    // Perform initial render to establish optimizer states
    auto render_output = performRendering(*mcmc);
    auto loss = render_output.image.mean();
    loss.backward();
    mcmc->step(1);

    // Get initial SH degree
    int initial_degree = mcmc->get_model().get_active_sh_degree();

    // Render again and call post_backward at iteration 1000
    render_output = performRendering(*mcmc);
    loss = render_output.image.mean();
    loss.backward();

    mcmc->post_backward(1000, render_output);

    // Check SH degree increased
    int new_degree = mcmc->get_model().get_active_sh_degree();
    EXPECT_EQ(new_degree, initial_degree + 1);
}

TEST_F(MCMCTest, GradientFlowTest) {
    // Test that gradients flow correctly through the full pipeline
    auto splat_data = createTestSplatData(50);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));
    mcmc->initialize(params.optimization);

    // Store initial values
    auto means_before = mcmc->get_model().means().clone();

    // Render and compute loss
    auto render_output = performRendering(*mcmc);
    auto loss = render_output.image.sum(); // Use sum for stronger gradients
    loss.backward();

    // Check gradients exist and are non-zero
    auto means_grad = mcmc->get_model().means().grad();
    EXPECT_TRUE(means_grad.defined());
    auto grad_norm = means_grad.norm();
    EXPECT_GT(grad_norm.item<float>(), 0) << "Gradients should be non-zero";

    // Take optimizer step
    mcmc->step(1);

    // Check that parameters changed
    auto means_after = mcmc->get_model().means();
    auto diff = (means_after - means_before).abs().sum();
    EXPECT_GT(diff.item<float>(), 0) << "Parameters should change after optimizer step";
}

TEST_F(MCMCTest, NoiseInjectionWithRenderingTest) {
    // Test noise injection in the context of actual rendering
    auto splat_data = createTestSplatData(20);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));
    mcmc->initialize(params.optimization);

    // Perform initial render and step to initialize optimizer
    auto render_output = performRendering(*mcmc);
    auto loss = render_output.image.mean();
    loss.backward();
    mcmc->step(1);

    // Store initial positions
    auto means_before = mcmc->get_model().means().clone();

    // Render again
    render_output = performRendering(*mcmc);
    loss = render_output.image.mean();
    loss.backward();

    // Call post_backward at iteration 100 (no refinement, just noise)
    mcmc->post_backward(100, render_output);

    // Positions should have changed due to noise injection
    auto means_after = mcmc->get_model().means();
    auto diff = (means_after - means_before).abs().sum();
    EXPECT_GT(diff.item<float>(), 0) << "Positions should change due to noise injection";
}

TEST_F(MCMCTest, RefinementWithActualRenderingTest) {
    // Test refinement (relocation and addition) with actual rendering
    auto splat_data = createTestSplatData(30);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));

    // Set up for refinement
    params.optimization.start_refine = 500;
    params.optimization.stop_refine = 1000;
    params.optimization.refine_every = 100;
    mcmc->initialize(params.optimization);

    // Initialize optimizer
    auto render_output = performRendering(*mcmc);
    auto loss = render_output.image.mean();
    loss.backward();
    mcmc->step(1);

    int initial_size = mcmc->get_model().size();

    // Run refinement step
    render_output = performRendering(*mcmc);
    loss = render_output.image.mean();
    loss.backward();

    mcmc->post_backward(600, render_output);

    // Size might have changed due to refinement
    int new_size = mcmc->get_model().size();
    std::cout << "Size changed from " << initial_size << " to " << new_size << std::endl;
}

TEST_F(MCMCTest, RelocationMechanicsTest) {
    // Detailed test of relocation with actual rendering
    auto splat_data = createTestSplatData(100);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));

    params.optimization.min_opacity = 0.005f;
    params.optimization.start_refine = 500;
    params.optimization.stop_refine= 1000;
    mcmc->initialize(params.optimization);

    // Initialize optimizer
    auto render_output = performRendering(*mcmc);
    auto loss = render_output.image.mean();
    loss.backward();
    mcmc->step(1);

    // Manually set some opacities to be very low
    {
        torch::NoGradGuard no_grad;
        auto opacity_raw = mcmc->get_model().opacity_raw();
        // Set first 20 Gaussians to have very low opacity
        opacity_raw.slice(0, 0, 20).fill_(torch::logit(torch::tensor(0.001f)));
    }

    // Store positions of low-opacity Gaussians
    auto low_opacity_means_before = mcmc->get_model().means().slice(0, 0, 20).clone();

    // Trigger relocation through rendering
    render_output = performRendering(*mcmc);
    loss = render_output.image.mean();
    loss.backward();

    mcmc->post_backward(600, render_output);

    // Check that low-opacity Gaussians were relocated (positions changed)
    auto low_opacity_means_after = mcmc->get_model().means().slice(0, 0, 20);
    auto position_diff = (low_opacity_means_after - low_opacity_means_before).abs().sum();
    EXPECT_GT(position_diff.item<float>(), 0) << "Low-opacity Gaussians should be relocated";

    // Check that all opacities are now above threshold
    auto opacities = mcmc->get_model().get_opacity();
    if (opacities.dim() == 2) {
        opacities = opacities.squeeze(-1);
    }
    auto min_opacity_value = opacities.min();
    EXPECT_GE(min_opacity_value.item<float>(), params.optimization.min_opacity * 0.9f)
        << "All opacities should be above threshold after relocation";
}

TEST_F(MCMCTest, ConsistentRenderingAfterOperationsTest) {
    // Test that rendering remains consistent after MCMC operations
    auto splat_data = createTestSplatData(50);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));
    mcmc->initialize(params.optimization);

    // Initial render
    auto render1 = performRendering(*mcmc);

    // Run through several iterations
    for (int iter = 1; iter <= 5; ++iter) {
        auto render_output = performRendering(*mcmc);
        auto loss = render_output.image.mean();
        loss.backward();

        mcmc->post_backward(500 + iter * 100, render_output);
        mcmc->step(500 + iter * 100);
    }

    // Final render
    auto render2 = performRendering(*mcmc);

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

TEST_F(MCMCTest, MultipleRefinementCyclesTest) {
    // Test multiple refinement cycles with actual rendering
    auto splat_data = createTestSplatData(50);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));

    params.optimization.max_cap = 200; // Low cap to test limits
    params.optimization.start_refine = 100;
    params.optimization.stop_refine = 1000;
    params.optimization.refine_every = 100;
    mcmc->initialize(params.optimization);

    // Track size changes
    std::vector<int> sizes;
    sizes.push_back(mcmc->get_model().size());

    // Run multiple refinement cycles
    for (int iter = 100; iter <= 500; iter += 100) {
        auto render_output = performRendering(*mcmc);
        auto loss = render_output.image.mean();
        loss.backward();

        mcmc->post_backward(iter, render_output);
        mcmc->step(iter);

        sizes.push_back(mcmc->get_model().size());

        // Verify size never exceeds cap
        EXPECT_LE(mcmc->get_model().size(), params.optimization.max_cap)
            << "Size should never exceed max_cap";
    }

    // Verify some growth happened
    EXPECT_GT(sizes.back(), sizes.front()) << "Some Gaussians should have been added";
}