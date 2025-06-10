#include "Ops.h"
#include "core/rasterizer_autograd.hpp"
#include "torch_impl.hpp"
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

// Using the exposed autograd functions from gs namespace
using gs::ProjectionFunction;
using gs::QuatScaleToCovarPreciFunction;
using gs::SphericalHarmonicsFunction;

class NumericalGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }

        device = torch::kCUDA;
        eps = 1e-4;

        // Clear any previous CUDA errors
        torch::cuda::synchronize();
        auto error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "Warning: Previous CUDA error cleared: " << cudaGetErrorString(error) << std::endl;
        }
    }

    torch::Tensor compute_numerical_gradient(
        std::function<torch::Tensor(torch::Tensor)> func,
        const torch::Tensor& input,
        float epsilon = 1e-4) {

        auto grad = torch::zeros_like(input);
        auto input_flat = input.view(-1);
        auto grad_flat = grad.view(-1);

        for (int64_t i = 0; i < input_flat.numel(); ++i) {
            float orig_val = input_flat[i].item<float>();

            input_flat[i] = orig_val + epsilon;
            auto f_plus = func(input).sum();

            input_flat[i] = orig_val - epsilon;
            auto f_minus = func(input).sum();

            grad_flat[i] = (f_plus.item<float>() - f_minus.item<float>()) / (2.0f * epsilon);

            input_flat[i] = orig_val;
        }

        return grad.view_as(input);
    }

    void compare_gradients(
        const torch::Tensor& analytical,
        const torch::Tensor& numerical,
        const std::string& name,
        float rtol = 1e-3,
        float atol = 1e-3) {

        auto diff = (analytical - numerical).abs();
        auto rel_diff = diff / (analytical.abs() + 1e-8);

        std::cout << "\n"
                  << name << " gradient check:" << std::endl;
        std::cout << "  Max absolute difference: " << diff.max().item<float>() << std::endl;
        std::cout << "  Mean absolute difference: " << diff.mean().item<float>() << std::endl;
        std::cout << "  Max relative difference: " << rel_diff.max().item<float>() << std::endl;
        std::cout << "  Mean relative difference: " << rel_diff.mean().item<float>() << std::endl;

        auto close = torch::allclose(analytical, numerical, rtol, atol);
        EXPECT_TRUE(close) << "Gradients do not match for " << name;

        EXPECT_FALSE(analytical.isnan().any().item<bool>()) << name << " has NaN analytical gradients";
        EXPECT_FALSE(analytical.isinf().any().item<bool>()) << name << " has Inf analytical gradients";
    }

    torch::Device device{torch::kCPU};
    float eps;
};

TEST_F(NumericalGradientTest, QuatScaleToCovarGradientTest) {
    torch::manual_seed(42);

    int N = 5;
    auto quats = torch::randn({N, 4}, device);
    quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto scales = torch::rand({N, 3}, device) * 0.1f + 0.01f;

    quats.requires_grad_(true);
    scales.requires_grad_(true);

    auto settings = torch::tensor({1.0f, 0.0f, 0.0f}, device); // compute_covar=true, compute_preci=false, triu=false

    // Test covariance gradients for quaternions
    {
        // Analytical gradient using autograd function
        auto outputs = QuatScaleToCovarPreciFunction::apply(quats, scales, settings);
        auto covars = outputs[0];

        // Use torch::autograd::grad like Python does
        auto v_covars = torch::randn_like(covars);
        auto grads = torch::autograd::grad(
            {(covars * v_covars).sum()},
            {quats},
            /*grad_outputs=*/{},
            /*retain_graph=*/false,
            /*create_graph=*/false);
        auto analytical_grad = grads[0];

        // Define function for numerical gradient
        auto covar_func = [&](torch::Tensor x) -> torch::Tensor {
            torch::NoGradGuard no_grad;
            auto [covars, _] = gsplat::quat_scale_to_covar_preci_fwd(
                x, scales.detach(), true, false, false);
            return (covars * v_covars).sum().unsqueeze(0);
        };

        // Numerical gradient
        auto numerical_grad = compute_numerical_gradient(covar_func, quats.detach(), eps);

        // Compare with Python tolerances
        compare_gradients(analytical_grad, numerical_grad, "quats->covars", 1e-1, 1e-1);
    }

    // Test scale gradients
    {
        auto outputs = QuatScaleToCovarPreciFunction::apply(quats, scales, settings);
        auto covars = outputs[0];

        auto v_covars = torch::randn_like(covars);
        auto grads = torch::autograd::grad(
            {(covars * v_covars).sum()},
            {scales},
            /*grad_outputs=*/{},
            /*retain_graph=*/false,
            /*create_graph=*/false);
        auto analytical_grad = grads[0];

        auto scale_func = [&](torch::Tensor x) -> torch::Tensor {
            torch::NoGradGuard no_grad;
            auto [covars, _] = gsplat::quat_scale_to_covar_preci_fwd(
                quats.detach(), x, true, false, false);
            return (covars * v_covars).sum().unsqueeze(0);
        };

        auto numerical_grad = compute_numerical_gradient(scale_func, scales.detach(), eps);

        compare_gradients(analytical_grad, numerical_grad, "scales->covars", 1e-1, 1e-1);
    }
}

TEST_F(NumericalGradientTest, SphericalHarmonicsGradientTest) {
    torch::manual_seed(42);

    // Test different SH degrees - exactly like Python
    std::vector<int> sh_degrees = {0, 1, 2, 3, 4};

    for (int sh_degree : sh_degrees) {
        std::cout << "Testing SH degree " << sh_degree << std::endl;

        int N = 1000;
        auto coeffs = torch::randn({N, (4 + 1) * (4 + 1), 3}, device);
        auto dirs = torch::randn({N, 3}, device);
        coeffs.requires_grad_(true);
        dirs.requires_grad_(true);

        // Create sh_degree tensor
        auto sh_degree_tensor = torch::tensor({sh_degree}, torch::TensorOptions().dtype(torch::kInt32).device(device));

        // CUDA implementation - directly call SphericalHarmonicsFunction::apply
        // Create masks with all ones since we want all Gaussians to be visible in the test
        auto masks = torch::ones({N}, torch::TensorOptions().dtype(torch::kBool).device(device));
        auto colors = SphericalHarmonicsFunction::apply(
            sh_degree_tensor, dirs, coeffs, masks)[0];

        // Reference implementation
        auto _colors = reference::spherical_harmonics(sh_degree, dirs, coeffs);

        // Forward should match
        EXPECT_TRUE(torch::allclose(colors, _colors, 1e-4, 1e-4))
            << "Forward pass mismatch for SH degree " << sh_degree;

        // Test gradients
        auto v_colors = torch::randn_like(colors);

        // CUDA gradients - fix the syntax for torch::autograd::grad
        auto loss = (colors * v_colors).sum();
        std::vector<torch::Tensor> inputs = {coeffs, dirs};
        std::vector<torch::Tensor> grad_outputs = {};

        auto grads = torch::autograd::grad(
            /*outputs=*/{loss},
            /*inputs=*/inputs,
            /*grad_outputs=*/grad_outputs,
            /*retain_graph=*/true,
            /*create_graph=*/false,
            /*allow_unused=*/true);
        auto v_coeffs = grads[0];
        auto v_dirs = grads[1];

        // Reference gradients
        auto _loss = (_colors * v_colors).sum();
        auto _grads = torch::autograd::grad(
            /*outputs=*/{_loss},
            /*inputs=*/inputs,
            /*grad_outputs=*/grad_outputs,
            /*retain_graph=*/true,
            /*create_graph=*/false,
            /*allow_unused=*/true);
        auto _v_coeffs = _grads[0];
        auto _v_dirs = _grads[1];

        // Check coefficient gradients
        EXPECT_TRUE(torch::allclose(v_coeffs, _v_coeffs, 1e-4, 1e-4))
            << "Coefficient gradients mismatch for SH degree " << sh_degree;

        // Check direction gradients (only for degree > 0)
        if (sh_degree > 0 && v_dirs.defined() && _v_dirs.defined()) {
            EXPECT_TRUE(torch::allclose(v_dirs, _v_dirs, 1e-4, 1e-4))
                << "Direction gradients mismatch for SH degree " << sh_degree;
        }
    }
}

TEST_F(NumericalGradientTest, ProjectionGradientTest) {
    // This test compares ProjectionFunction against reference implementation
    // Both now support optional opacities - pass undefined to match behavior
    torch::manual_seed(42);

    // Test data
    int N = 100;
    int C = 1;
    int width = 256;
    int height = 256;
    float eps2d = 0.3f;
    float near_plane = 0.01f;
    float far_plane = 10000.0f;
    float radius_clip = 0.0f;
    float scaling_modifier = 1.0f;
    bool calc_compensations = false; // C++ version always returns compensations but set to 1.0

    // Create test data
    auto means3D = torch::randn({N, 3}, device) * 5.0;
    auto quats = torch::randn({N, 4}, device);
    auto scales = torch::rand({N, 3}, device) * 0.5;
    // Don't create opacities - pass undefined tensor to match reference

    // Camera parameters
    auto viewmat = torch::eye(4, device).unsqueeze(0);
    viewmat.index_put_({0, 2, 3}, 10.0f); // Move camera back

    auto K = torch::tensor({{width / 2.0f, 0.0f, width / 2.0f},
                            {0.0f, height / 2.0f, height / 2.0f},
                            {0.0f, 0.0f, 1.0f}},
                           device)
                 .unsqueeze(0);

    // Set requires_grad
    means3D.requires_grad_(true);
    quats.requires_grad_(true);
    scales.requires_grad_(true);
    // Don't set requires_grad for opacities since reference doesn't use them
    viewmat.requires_grad_(true);

    // Pack settings for C++ version
    auto settings = torch::tensor({(float)width,
                                   (float)height,
                                   eps2d,
                                   near_plane,
                                   far_plane,
                                   radius_clip,
                                   scaling_modifier},
                                  torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // C++ implementation - use opacities=1.0 to disable opacity-based optimization
    // This makes the behavior equivalent to not using opacities
    auto opacities = torch::ones({N}, device);
    auto proj_outputs = ProjectionFunction::apply(
        means3D, quats, scales, opacities, viewmat, K, settings);

    auto radii = proj_outputs[0];
    auto means2d = proj_outputs[1];
    auto depths = proj_outputs[2];
    auto conics = proj_outputs[3];
    auto compensations = proj_outputs[4];

    // Reference implementation using quat_scale_to_covar_preci first
    // IMPORTANT: Apply scaling_modifier to scales just like C++ implementation does
    auto scaled_scales = scales * scaling_modifier;
    auto covar_settings = torch::tensor({1.0f, 0.0f, 0.0f}, device); // compute_covar=true, compute_preci=false, triu=false
    auto covar_outputs = QuatScaleToCovarPreciFunction::apply(quats, scaled_scales, covar_settings);
    auto covars = covar_outputs[0]; // [N, 3, 3]

    // Call reference projection
    // Note: reference implementation expects camera_model as string parameter
    auto ref_outputs = reference::fully_fused_projection(
        means3D, covars, viewmat, K, width, height, eps2d, near_plane, far_plane,
        false,    // calc_compensations
        "pinhole" // camera_model
    );

    auto _radii = std::get<0>(ref_outputs);
    auto _means2d = std::get<1>(ref_outputs);
    auto _depths = std::get<2>(ref_outputs);
    auto _conics = std::get<3>(ref_outputs);
    auto _compensations = std::get<4>(ref_outputs);

    // Note: When calc_compensations=false, reference returns undefined tensor
    // but C++ version always returns defined tensor (ones)
    if (!_compensations.defined()) {
        _compensations = torch::ones({C, N}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }

    // Forward pass checks
    auto valid = (radii > 0).all(-1) & (_radii > 0).all(-1);
    valid = valid.squeeze(0); // Remove camera dimension

    // radii is integer so we allow for 1 unit difference
    EXPECT_TRUE(torch::allclose(radii, _radii, 0, 1))
        << "Radii mismatch";

    // Check other outputs only for valid Gaussians
    if (valid.any().item().toBool()) {
        auto valid_mask = valid.unsqueeze(0).unsqueeze(-1);
        EXPECT_TRUE(torch::allclose(
            means2d.masked_select(valid_mask.expand_as(means2d)),
            _means2d.masked_select(valid_mask.expand_as(_means2d)),
            1e-4, 1e-4))
            << "Means2D mismatch for valid Gaussians";

        auto valid_mask_1d = valid.unsqueeze(0);
        EXPECT_TRUE(torch::allclose(
            depths.masked_select(valid_mask_1d),
            _depths.masked_select(valid_mask_1d),
            1e-4, 1e-4))
            << "Depths mismatch for valid Gaussians";

        EXPECT_TRUE(torch::allclose(
            conics.masked_select(valid_mask.expand_as(conics)),
            _conics.masked_select(valid_mask.expand_as(_conics)),
            1e-4, 1e-4))
            << "Conics mismatch for valid Gaussians";
    }

    // Backward pass test
    auto v_means2d = torch::randn_like(means2d) * valid.unsqueeze(0).unsqueeze(-1);
    auto v_depths = torch::randn_like(depths) * valid.unsqueeze(0);
    auto v_conics = torch::randn_like(conics) * valid.unsqueeze(0).unsqueeze(-1);
    auto v_compensations = torch::zeros_like(compensations); // No gradient for compensations in this test

    // Compute loss for C++ version
    auto loss = (means2d * v_means2d).sum() +
                (depths * v_depths).sum() +
                (conics * v_conics).sum();

    // Get gradients
    // Note: Don't include opacities in inputs since reference doesn't use them
    std::vector<torch::Tensor> inputs = {means3D, quats, scales, viewmat};
    std::vector<torch::Tensor> outputs = {loss};
    std::vector<torch::Tensor> grad_outputs = {};

    auto grads = torch::autograd::grad(
        outputs,
        inputs,
        grad_outputs,
        /*retain_graph=*/true,
        /*create_graph=*/false,
        /*allow_unused=*/true);

    auto v_means3D = grads[0];
    auto v_quats = grads[1];
    auto v_scales = grads[2];
    auto v_viewmat = grads[3];

    // Reference gradients - note that reference doesn't use opacities
    std::vector<torch::Tensor> ref_inputs = {means3D, quats, scales, viewmat};
    auto _loss = (_means2d * v_means2d).sum() +
                 (_depths * v_depths).sum() +
                 (_conics * v_conics).sum();
    std::vector<torch::Tensor> ref_outputs_grad = {_loss};

    auto _grads = torch::autograd::grad(
        ref_outputs_grad,
        ref_inputs,
        grad_outputs,
        /*retain_graph=*/true,
        /*create_graph=*/false,
        /*allow_unused=*/true);

    auto _v_means3D = _grads[0];
    auto _v_quats = _grads[1];
    auto _v_scales = _grads[2];
    auto _v_viewmat = _grads[3];

    // Check gradients with relaxed tolerances as in Python test
    EXPECT_TRUE(torch::allclose(v_viewmat, _v_viewmat, 2e-3, 2e-3))
        << "Viewmat gradients mismatch\nMax diff: " << (v_viewmat - _v_viewmat).abs().max().item().toFloat();

    EXPECT_TRUE(torch::allclose(v_quats, _v_quats, 2e-1, 2e-2))
        << "Quaternion gradients mismatch\nMax diff: " << (v_quats - _v_quats).abs().max().item().toFloat();

    EXPECT_TRUE(torch::allclose(v_scales, _v_scales, 1e-1, 2e-1))
        << "Scale gradients mismatch\nMax diff: " << (v_scales - _v_scales).abs().max().item().toFloat();

    EXPECT_TRUE(torch::allclose(v_means3D, _v_means3D, 1e-2, 6e-2))
        << "Means3D gradients mismatch\nMax diff: " << (v_means3D - _v_means3D).abs().max().item().toFloat();
}

TEST_F(NumericalGradientTest, CompareWithReferenceImplementation) {
    torch::manual_seed(42);

    // Test quat_scale_to_covar_preci against reference
    {
        int N = 10;
        auto quats = torch::randn({N, 4}, device);
        quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
        auto scales = torch::rand({N, 3}, device) * 0.1f;

        // gsplat implementation
        auto [covars, precis] = gsplat::quat_scale_to_covar_preci_fwd(
            quats, scales, true, true, false);

        // Reference implementation
        auto [ref_covars, ref_precis] = reference::quat_scale_to_covar_preci(
            quats, scales, true, true, false);

        // Compare
        EXPECT_TRUE(torch::allclose(covars, ref_covars, 1e-5, 1e-5))
            << "Covariance matrices don't match reference";
        auto preci_diff = (precis - ref_precis).abs().max();
        std::cout << "Max precision matrix difference: " << preci_diff.item<float>() << std::endl;
    }

    // Test spherical harmonics against reference
    {
        int N = 100;
        int degree = 2;
        int K = (degree + 1) * (degree + 1);

        auto dirs = torch::randn({N, 3}, device);
        auto coeffs = torch::randn({N, K, 3}, device);
        auto masks = torch::ones({N}, torch::TensorOptions().dtype(torch::kBool).device(device));

        // gsplat implementation
        auto colors = gsplat::spherical_harmonics_fwd(degree, dirs, coeffs, masks);

        // Reference implementation
        auto ref_colors = reference::spherical_harmonics(degree, dirs, coeffs);

        // Compare
        EXPECT_TRUE(torch::allclose(colors, ref_colors, 1e-4, 1e-4))
            << "Spherical harmonics don't match reference";
    }
}

TEST_F(NumericalGradientTest, StressTestGradients) {
    torch::manual_seed(42);

    // Very small scales
    {
        int N = 5;
        auto quats = torch::randn({N, 4}, device);
        quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
        auto scales = torch::ones({N, 3}, device) * 1e-6f;

        quats.requires_grad_(true);
        scales.requires_grad_(true);

        auto settings = torch::tensor({1.0f, 1.0f, 0.0f}, device);

        auto outputs = QuatScaleToCovarPreciFunction::apply(quats, scales, settings);
        auto covars = outputs[0];
        auto precis = outputs[1];

        auto v_covars = torch::randn_like(covars);
        auto v_precis = torch::randn_like(precis) * 1e-6f;

        auto grads = torch::autograd::grad(
            {(covars * v_covars).sum() + (precis * v_precis).sum()},
            {quats, scales},
            /*grad_outputs=*/{},
            /*retain_graph=*/false,
            /*create_graph=*/false);

        auto quats_grad_finite = torch::isfinite(grads[0]).all().item<bool>();
        auto scales_grad_finite = torch::isfinite(grads[1]).all().item<bool>();

        EXPECT_TRUE(quats_grad_finite)
            << "Non-finite gradients with small scales";
        EXPECT_TRUE(scales_grad_finite)
            << "Non-finite gradients with small scales";
    }

    // Near-zero quaternions (should be normalized)
    {
        int N = 5;
        auto quats = torch::randn({N, 4}, device) * 0.01f;
        quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
        auto scales = torch::rand({N, 3}, device) * 0.1f;

        quats.requires_grad_(true);
        scales.requires_grad_(true);

        auto settings = torch::tensor({1.0f, 0.0f, 0.0f}, device);

        auto outputs = QuatScaleToCovarPreciFunction::apply(quats, scales, settings);
        auto covars = outputs[0];

        auto v_covars = torch::randn_like(covars);
        auto grads = torch::autograd::grad(
            {(covars * v_covars).sum()},
            {quats},
            /*grad_outputs=*/{},
            /*retain_graph=*/false,
            /*create_graph=*/false);

        auto quats_grad_finite = torch::isfinite(grads[0]).all().item<bool>();

        EXPECT_TRUE(quats_grad_finite)
            << "Non-finite gradients with near-zero quaternions";
    }
}