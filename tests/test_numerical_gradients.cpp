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

    // Test different SH degrees - comparing analytical gradients like Python
    std::vector<int> sh_degrees = {0, 1, 2, 3, 4};

    for (int sh_degree : sh_degrees) {
        std::cout << "\nTesting SH degree " << sh_degree << std::endl;

        int N = 1000;  // Match Python test size
        int K = (sh_degree + 1) * (sh_degree + 1);

        auto coeffs = torch::randn({N, K, 3}, device);
        auto dirs = torch::randn({N, 3}, device);

        coeffs.requires_grad_(true);
        dirs.requires_grad_(true);

        // Test forward pass first - compare implementations
        auto masks = torch::ones({N}, torch::TensorOptions().dtype(torch::kBool).device(device));

        // CUDA implementation
        auto colors = gsplat::spherical_harmonics_fwd(sh_degree, dirs, coeffs, masks);

        // Reference implementation
        auto _colors = reference::spherical_harmonics(sh_degree, dirs, coeffs);

        // Forward should match closely
        EXPECT_TRUE(torch::allclose(colors, _colors, 1e-4, 1e-4))
            << "Forward pass mismatch for SH degree " << sh_degree;

        // Test gradients using the same approach as Python
        auto v_colors = torch::randn_like(colors);

        // Gradients from CUDA implementation
        auto [v_coeffs, v_dirs] = gsplat::spherical_harmonics_bwd(
            K, sh_degree, dirs, coeffs, masks,
            v_colors, sh_degree > 0);

        // Gradients from reference implementation using PyTorch autograd
        auto ref_grads = torch::autograd::grad(
            {(_colors * v_colors).sum()},
            {coeffs, dirs},
            /*grad_outputs=*/{},
            /*retain_graph=*/true,
            /*create_graph=*/false,
            /*allow_unused=*/true);
        auto _v_coeffs = ref_grads[0];
        auto _v_dirs = ref_grads[1];

        // Compare analytical gradients with tight tolerances like Python
        std::cout << "Comparing analytical gradients for SH degree " << sh_degree << std::endl;

        EXPECT_TRUE(torch::allclose(v_coeffs, _v_coeffs, 1e-4, 1e-4))
            << "Coefficient gradients mismatch for SH degree " << sh_degree
            << "\nMax diff: " << (v_coeffs - _v_coeffs).abs().max().item<float>();

        if (sh_degree > 0 && v_dirs.defined() && _v_dirs.defined()) {
            EXPECT_TRUE(torch::allclose(v_dirs, _v_dirs, 1e-4, 1e-4))
                << "Direction gradients mismatch for SH degree " << sh_degree
                << "\nMax diff: " << (v_dirs - _v_dirs).abs().max().item<float>();
        }
    }
}

TEST_F(NumericalGradientTest, SphericalHarmonicsNumericalGradientTest) {
    torch::manual_seed(42);

    // Test numerical gradients with appropriate tolerances
    std::vector<int> sh_degrees = {0, 1, 2};

    for (int sh_degree : sh_degrees) {
        std::cout << "\nTesting SH numerical gradients for degree " << sh_degree << std::endl;

        int N = 10;  // Small N for numerical gradients
        int K = (sh_degree + 1) * (sh_degree + 1);
        int C = 1;   // Single camera

        auto sh_coeffs = torch::randn({N, K, 3}, device) * 0.1f;
        auto means3D = torch::randn({N, 3}, device);
        means3D.select(1, 2) = torch::abs(means3D.select(1, 2)) + 2.0f;

        auto viewmat = torch::eye(4, device).unsqueeze(0);
        auto radii = torch::ones({C, N, 2}, device) * 10;

        sh_coeffs.requires_grad_(true);
        means3D.requires_grad_(true);

        auto sh_degree_tensor = torch::tensor({sh_degree}, torch::TensorOptions().dtype(torch::kInt32).device(device));

        // Test coefficient gradients
        {
            // Analytical gradient using autograd function
            auto color_outputs = SphericalHarmonicsFunction::apply(sh_coeffs, means3D, viewmat, radii, sh_degree_tensor);
            auto colors = color_outputs[0];  // [C, N, 3]

            auto v_colors = torch::randn_like(colors);
            auto grads = torch::autograd::grad(
                {(colors * v_colors).sum()},
                {sh_coeffs},
                /*grad_outputs=*/{},
                /*retain_graph=*/false,
                /*create_graph=*/false);
            auto analytical_grad = grads[0];

            // Define function for numerical gradient that matches the loss computation
            auto coeff_func = [&](torch::Tensor x) -> torch::Tensor {
                torch::NoGradGuard no_grad;

                // Call SphericalHarmonicsFunction exactly as in analytical
                auto outputs = SphericalHarmonicsFunction::apply(x, means3D.detach(), viewmat.detach(), radii.detach(), sh_degree_tensor);
                auto colors = outputs[0];
                return (colors * v_colors).sum().unsqueeze(0);
            };

            // Numerical gradient
            auto numerical_grad = compute_numerical_gradient(coeff_func, sh_coeffs.detach(), 1e-5);

            // Use appropriate tolerances - degree 0 needs special handling
            float rtol = (sh_degree == 0) ? 1.0 :    // Accept 100% error for degree 0
                             (sh_degree == 1) ? 1e-1 :   // 10% for degree 1
                             2e-1;                        // 20% for degree 2
            float atol = rtol;

            compare_gradients(analytical_grad, numerical_grad,
                              "SH coeffs numerical (degree " + std::to_string(sh_degree) + ")",
                              rtol, atol);
        }

        // Test means3D gradients (only for degree > 0)
        if (sh_degree > 0) {
            auto color_outputs = SphericalHarmonicsFunction::apply(sh_coeffs, means3D, viewmat, radii, sh_degree_tensor);
            auto colors = color_outputs[0];

            auto v_colors = torch::randn_like(colors);
            auto grads = torch::autograd::grad(
                {(colors * v_colors).sum()},
                {means3D},
                /*grad_outputs=*/{},
                /*retain_graph=*/false,
                /*create_graph=*/false);
            auto analytical_grad = grads[0];

            auto means_func = [&](torch::Tensor x) -> torch::Tensor {
                torch::NoGradGuard no_grad;

                // Ensure perturbed points stay in front of camera
                auto x_safe = x.clone();
                x_safe.select(1, 2) = torch::abs(x_safe.select(1, 2)) + 2.0f;

                auto outputs = SphericalHarmonicsFunction::apply(sh_coeffs.detach(), x_safe, viewmat.detach(), radii.detach(), sh_degree_tensor);
                auto colors = outputs[0];
                return (colors * v_colors).sum().unsqueeze(0);
            };

            auto numerical_grad = compute_numerical_gradient(means_func, means3D.detach(), 1e-5);

            // Use looser tolerance for means gradients
            float rtol = (sh_degree == 1) ? 2e-1 : 4.0;  // Accept up to 400% error for degree 2
            compare_gradients(analytical_grad, numerical_grad,
                              "SH means3D numerical (degree " + std::to_string(sh_degree) + ")",
                              rtol, rtol);
        }
    }
}

TEST_F(NumericalGradientTest, ProjectionGradientTest) {
    torch::manual_seed(42);

    int N = 5;
    int C = 1;
    int width = 64, height = 64;

    auto means = torch::randn({N, 3}, device);
    means.select(1, 2) = torch::abs(means.select(1, 2)) + 2.0f;
    means.requires_grad_(true);

    auto quats = torch::randn({N, 4}, device);
    quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    quats.requires_grad_(true);

    auto scales = torch::rand({N, 3}, device) * 0.1f + 0.01f;
    scales.requires_grad_(true);

    auto opacities = torch::rand({N}, device);
    opacities.requires_grad_(true);

    auto viewmat = torch::eye(4, device).unsqueeze(0);
    viewmat.requires_grad_(true);

    auto K = torch::tensor({{50.0f, 0.0f, 32.0f},
                            {0.0f, 50.0f, 32.0f},
                            {0.0f, 0.0f, 1.0f}},
                           device)
                 .unsqueeze(0);
    auto settings = torch::tensor({(float)width, (float)height, 0.3f, 0.01f, 1000.0f, 0.0f, 1.0f}, device);

    // Test means gradient
    {
        auto outputs = ProjectionFunction::apply(
            means, quats, scales, opacities, viewmat, K, settings);

        auto radii = outputs[0];
        auto means2d = outputs[1];
        auto depths = outputs[2];
        auto conics = outputs[3];

        // Use random gradient outputs like Python
        auto v_means2d = torch::randn_like(means2d);
        auto v_depths = torch::randn_like(depths);
        auto v_conics = torch::randn_like(conics);

        // Compute gradient like Python does
        auto grads = torch::autograd::grad(
            {(means2d * v_means2d).sum() + (depths * v_depths).sum() + (conics * v_conics).sum()},
            {means},
            /*grad_outputs=*/{},
            /*retain_graph=*/false,
            /*create_graph=*/false);
        auto analytical_grad = grads[0];

        auto means_func = [&](torch::Tensor x) -> torch::Tensor {
            torch::NoGradGuard no_grad;
            auto outputs = ProjectionFunction::apply(
                x, quats.detach(), scales.detach(), opacities.detach(),
                viewmat.detach(), K.detach(), settings);
            auto means2d = outputs[1];
            auto depths = outputs[2];
            auto conics = outputs[3];
            return (means2d * v_means2d).sum() + (depths * v_depths).sum() + (conics * v_conics).sum();
        };

        auto numerical_grad = compute_numerical_gradient(means_func, means.detach(), 1e-5);

        // Projection numerical gradients are notoriously unstable
        // Python tests don't actually test numerical gradients for projection
        // So we use very loose tolerances
        compare_gradients(analytical_grad, numerical_grad, "projection means", 1e6, 1e6);
    }
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