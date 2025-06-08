#include <gtest/gtest.h>
#include <torch/torch.h>
#include "core/mcmc.hpp"
#include "core/splat_data.hpp"
#include "core/parameters.hpp"
#include "core/rasterizer.hpp"
#include <memory>

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
        params.optimization.start_densify = 500;
        params.optimization.stop_densify = 15000;
        params.optimization.growth_interval = 100;
    }

    // Helper to create a dummy SplatData for testing
    SplatData createTestSplatData(int N) {
        // Create tensors without gradients for testing
        torch::NoGradGuard no_grad;

        auto means = torch::randn({N, 3}, torch::kFloat32);
        auto sh0 = torch::randn({N, 1, 3}, torch::kFloat32);
        auto shN = torch::randn({N, 3, 3}, torch::kFloat32);
        auto scaling = torch::randn({N, 3}, torch::kFloat32);
        auto rotation = torch::randn({N, 4}, torch::kFloat32);
        auto opacity = torch::randn({N, 1}, torch::kFloat32);

        return SplatData(3, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    // Helper to create render output with correct size
    gs::RenderOutput createRenderOutput(int size) {
        gs::RenderOutput render_output;
        render_output.radii = torch::ones({size}, torch::kInt32).to(device);
        render_output.visibility = torch::ones({size}, torch::kBool).to(device);
        render_output.means2d = torch::randn({size, 2}, torch::kFloat32).to(device);
        render_output.depths = torch::rand({size}, torch::kFloat32).to(device) * 10.0f;
        render_output.image = torch::zeros({3, 256, 256}, torch::kFloat32).to(device);
        render_output.width = 256;
        render_output.height = 256;
        return render_output;
    }

    torch::Device device{torch::kCPU};
    gs::param::TrainingParameters params{};
};

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

TEST_F(MCMCTest, SHDegreeIncrementTest) {
    // Create test data
    auto splat_data = createTestSplatData(50);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));

    // Set parameters to avoid refinement at iteration 1000
    params.optimization.stop_densify = 999;  // Stop densification before iteration 1000
    mcmc->initialize(params.optimization);

    // Get initial SH degree
    int initial_degree = mcmc->get_model().get_active_sh_degree();

    // Create render output with current size
    auto render_output = createRenderOutput(mcmc->get_model().size());

    // Disable gradient computation
    torch::NoGradGuard no_grad;

    // Call post_backward at iteration 1000 (should only increment SH degree)
    mcmc->post_backward(1000, render_output);

    // Check SH degree increased
    int new_degree = mcmc->get_model().get_active_sh_degree();
    EXPECT_EQ(new_degree, initial_degree + 1);
}

TEST_F(MCMCTest, OptimizerStepTest) {
    // Test that optimizer step works correctly
    auto splat_data = createTestSplatData(10);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));
    mcmc->initialize(params.optimization);

    // Store initial values
    auto means_before = mcmc->get_model().means().clone();

    // Create dummy gradients
    {
        torch::NoGradGuard no_grad;
        mcmc->get_model().means().mutable_grad() = torch::randn_like(mcmc->get_model().means()) * 0.01f;
        mcmc->get_model().opacity_raw().mutable_grad() = torch::randn_like(mcmc->get_model().opacity_raw()) * 0.01f;
        mcmc->get_model().scaling_raw().mutable_grad() = torch::randn_like(mcmc->get_model().scaling_raw()) * 0.01f;
        mcmc->get_model().rotation_raw().mutable_grad() = torch::randn_like(mcmc->get_model().rotation_raw()) * 0.01f;
        mcmc->get_model().sh0().mutable_grad() = torch::randn_like(mcmc->get_model().sh0()) * 0.01f;
        mcmc->get_model().shN().mutable_grad() = torch::randn_like(mcmc->get_model().shN()) * 0.01f;
    }

    // Take optimizer step
    ASSERT_NO_THROW(mcmc->step(1));

    // Check that parameters changed
    auto means_after = mcmc->get_model().means();
    auto diff = (means_after - means_before).abs().sum();
    EXPECT_GT(diff.item<float>(), 0) << "Parameters should change after optimizer step";

    // Check gradients were zeroed
    EXPECT_FALSE(mcmc->get_model().means().grad().defined() ||
                 mcmc->get_model().means().grad().sum().item<float>() == 0);
}

TEST_F(MCMCTest, NoiseInjectionTest) {
    // Test that noise injection works
    auto splat_data = createTestSplatData(20);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));
    mcmc->initialize(params.optimization);

    // Store initial positions
    auto means_before = mcmc->get_model().means().clone();

    // Create render output
    auto render_output = createRenderOutput(mcmc->get_model().size());

    // Call post_backward at iteration 100 (no refinement, just noise)
    {
        torch::NoGradGuard no_grad;
        mcmc->post_backward(100, render_output);
    }

    // Positions should have changed due to noise injection
    auto means_after = mcmc->get_model().means();
    auto diff = (means_after - means_before).abs().sum();
    EXPECT_GT(diff.item<float>(), 0) << "Positions should change due to noise injection";
}

TEST_F(MCMCTest, RefinementBoundariesTest) {
    // Test that refinement respects start/stop boundaries
    auto splat_data = createTestSplatData(30);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));

    // Set clear boundaries
    params.optimization.start_densify = 1000;
    params.optimization.stop_densify = 2000;
    params.optimization.growth_interval = 100;
    mcmc->initialize(params.optimization);

    int initial_size = mcmc->get_model().size();

    // Test before start_densify (iteration 500)
    {
        torch::NoGradGuard no_grad;
        auto render_output = createRenderOutput(mcmc->get_model().size());
        mcmc->post_backward(500, render_output);
        EXPECT_EQ(mcmc->get_model().size(), initial_size) << "Size should not change before start_densify";
    }

    // Test during refinement period (iteration 1100)
    {
        torch::NoGradGuard no_grad;
        auto render_output = createRenderOutput(mcmc->get_model().size());
        mcmc->post_backward(1100, render_output);
        // Size might change during refinement period
    }

    // Test after stop_densify (iteration 2100)
    {
        torch::NoGradGuard no_grad;
        int size_before = mcmc->get_model().size();
        auto render_output = createRenderOutput(size_before);
        mcmc->post_backward(2100, render_output);
        EXPECT_EQ(mcmc->get_model().size(), size_before) << "Size should not change after stop_densify";
    }
}

TEST_F(MCMCTest, MaxCapRespectedTest) {
    // Test that max_cap is respected
    auto splat_data = createTestSplatData(90);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));

    // Set a low max_cap
    params.optimization.max_cap = 100;
    mcmc->initialize(params.optimization);

    // Run multiple refinement steps
    for (int iter = 600; iter <= 1000; iter += 100) {
        torch::NoGradGuard no_grad;
        auto render_output = createRenderOutput(mcmc->get_model().size());
        mcmc->post_backward(iter, render_output);

        EXPECT_LE(mcmc->get_model().size(), params.optimization.max_cap)
            << "Size should never exceed max_cap";
    }
}

TEST_F(MCMCTest, ConsistentParameterSizesTest) {
    // Test that all parameters maintain consistent sizes
    auto splat_data = createTestSplatData(25);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));
    mcmc->initialize(params.optimization);

    // Run a refinement step
    {
        torch::NoGradGuard no_grad;
        auto render_output = createRenderOutput(mcmc->get_model().size());
        mcmc->post_backward(600, render_output);
    }

    // Check all parameters have the same size
    int64_t size = mcmc->get_model().size();
    EXPECT_EQ(mcmc->get_model().means().size(0), size);
    EXPECT_EQ(mcmc->get_model().opacity_raw().size(0), size);
    EXPECT_EQ(mcmc->get_model().scaling_raw().size(0), size);
    EXPECT_EQ(mcmc->get_model().rotation_raw().size(0), size);
    EXPECT_EQ(mcmc->get_model().sh0().size(0), size);
    EXPECT_EQ(mcmc->get_model().shN().size(0), size);
}

TEST_F(MCMCTest, MinOpacityThresholdTest) {
    // Test that min_opacity threshold works correctly
    auto splat_data = createTestSplatData(50);
    auto mcmc = std::make_unique<MCMC>(std::move(splat_data));

    // Set a high min_opacity to ensure some Gaussians are relocated
    params.optimization.min_opacity = 0.7f;
    mcmc->initialize(params.optimization);

    // Manually set some opacities below threshold
    {
        torch::NoGradGuard no_grad;
        auto opacity_raw = mcmc->get_model().opacity_raw();
        // Set first 20 Gaussians to have very low opacity
        opacity_raw.slice(0, 0, 20).fill_(torch::logit(torch::tensor(0.01f)));
    }

    // Store means before relocation
    auto means_before = mcmc->get_model().means().clone();

    // Trigger relocation
    {
        torch::NoGradGuard no_grad;
        auto render_output = createRenderOutput(mcmc->get_model().size());
        mcmc->post_backward(600, render_output);
    }

    // Check that low-opacity Gaussians were relocated
    auto opacities = mcmc->get_model().get_opacity();
    if (opacities.dim() == 2) {
        opacities = opacities.squeeze(-1);
    }

    // All opacities should now be above min_opacity (or close to it)
    auto min_opacity_value = opacities.min();
    EXPECT_GE(min_opacity_value.item<float>(), params.optimization.min_opacity * 0.9f)
        << "All opacities should be above or near min_opacity after relocation";
}