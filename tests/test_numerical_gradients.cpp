#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Ops.h"
#include "torch_impl.hpp"
#include "core/rasterizer_autograd.hpp"
#include <functional>
#include <vector>
#include <cuda_runtime.h>

// Using the exposed autograd functions from gs namespace
using gs::QuatScaleToCovarPreciFunction;
using gs::ProjectionFunction;
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

        std::cout << "\n" << name << " gradient check:" << std::endl;
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
        auto loss = covars.sum();
        loss.backward();
        auto analytical_grad = quats.grad().clone();

        // Clear gradients
        quats.grad().zero_();

        // Define function for numerical gradient
        auto covar_func = [&](torch::Tensor x) -> torch::Tensor {
            torch::NoGradGuard no_grad;
            auto [covars, _] = gsplat::quat_scale_to_covar_preci_fwd(
                x, scales.detach(), true, false, false);
            return covars;
        };

        // Numerical gradient
        auto numerical_grad = compute_numerical_gradient(covar_func, quats.detach(), eps);

        // Compare
        compare_gradients(analytical_grad, numerical_grad, "quats->covars");
    }

    // Clear gradients
    if (quats.grad().defined()) quats.grad().zero_();
    if (scales.grad().defined()) scales.grad().zero_();

    // Test scale gradients
    {
        auto scale_func = [&](torch::Tensor x) -> torch::Tensor {
            torch::NoGradGuard no_grad;
            auto [covars, _] = gsplat::quat_scale_to_covar_preci_fwd(
                quats.detach(), x, true, false, false);
            return covars;
        };

        auto outputs = QuatScaleToCovarPreciFunction::apply(quats, scales, settings);
        auto covars = outputs[0];
        auto loss = covars.sum();
        loss.backward();
        auto analytical_grad = scales.grad().clone();

        auto numerical_grad = compute_numerical_gradient(scale_func, scales.detach(), eps);

        compare_gradients(analytical_grad, numerical_grad, "scales->covars");
    }
}

TEST_F(NumericalGradientTest, SphericalHarmonicsGradientTest) {
    torch::manual_seed(42);

    // Test different SH degrees
    std::vector<int> sh_degrees = {0, 1, 2};

    for (int sh_degree : sh_degrees) {
        std::cout << "\nTesting SH degree " << sh_degree << std::endl;

        int N = 10;
        int K = (sh_degree + 1) * (sh_degree + 1);
        int C = 1; // Single camera

        auto sh_coeffs = torch::randn({N, K, 3}, device);
        auto means3D = torch::randn({N, 3}, device);
        auto viewmat = torch::eye(4, device).unsqueeze(0);
        auto radii = torch::ones({C, N, 2}, device) * 10; // All visible

        sh_coeffs.requires_grad_(true);
        means3D.requires_grad_(true);
        viewmat.requires_grad_(true);

        auto sh_degree_tensor = torch::tensor({sh_degree}, torch::TensorOptions().dtype(torch::kInt32).device(device));

        // Test coefficient gradients
        {
            auto coeff_func = [&](torch::Tensor x) -> torch::Tensor {
                torch::NoGradGuard no_grad;
                // Use the actual SphericalHarmonicsFunction interface
                auto viewmat_inv = torch::inverse(viewmat);
                auto campos = viewmat_inv.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 3});
                auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1);
                auto masks = (radii > 0).all(-1);

                if (sh_degree > 0 && x.size(1) > 1) {
                    int num_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
                    auto sh_coeffs_used = x.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, num_sh_coeffs), torch::indexing::Slice()});
                    auto colors = gsplat::spherical_harmonics_fwd(sh_degree, dirs[0], sh_coeffs_used, masks[0]);
                    return torch::clamp_min(colors + 0.5f, 0.0f);
                } else {
                    auto colors = x.index({torch::indexing::Slice(), 0, torch::indexing::Slice()});
                    return torch::clamp_min(colors + 0.5f, 0.0f);
                }
            };

            auto color_outputs = SphericalHarmonicsFunction::apply(sh_coeffs, means3D, viewmat, radii, sh_degree_tensor);
            auto colors = color_outputs[0]; // Extract tensor from tensor_list
            auto loss = colors.sum();
            loss.backward();
            auto analytical_grad = sh_coeffs.grad().clone();

            auto numerical_grad = compute_numerical_gradient(coeff_func, sh_coeffs.detach(), eps);

            compare_gradients(analytical_grad, numerical_grad,
                              "SH coeffs (degree " + std::to_string(sh_degree) + ")");
        }

        // Clear gradients
        sh_coeffs.grad().zero_();
        if (means3D.grad().defined()) means3D.grad().zero_();
        if (viewmat.grad().defined()) viewmat.grad().zero_();

        // Test means3D gradients (only for degree > 0)
        if (sh_degree > 0) {
            auto means_func = [&](torch::Tensor x) -> torch::Tensor {
                torch::NoGradGuard no_grad;
                auto viewmat_inv = torch::inverse(viewmat);
                auto campos = viewmat_inv.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 3});
                auto dirs = x.unsqueeze(0) - campos.unsqueeze(1);
                auto masks = (radii > 0).all(-1);

                int num_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
                auto sh_coeffs_used = sh_coeffs.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, num_sh_coeffs), torch::indexing::Slice()});
                auto colors = gsplat::spherical_harmonics_fwd(sh_degree, dirs[0], sh_coeffs_used.detach(), masks[0]);
                return torch::clamp_min(colors + 0.5f, 0.0f);
            };

            auto color_outputs = SphericalHarmonicsFunction::apply(sh_coeffs, means3D, viewmat, radii, sh_degree_tensor);
            auto colors = color_outputs[0]; // Extract tensor from tensor_list
            auto loss = colors.sum();
            loss.backward();
            auto analytical_grad = means3D.grad().clone();

            auto numerical_grad = compute_numerical_gradient(means_func, means3D.detach(), eps);

            compare_gradients(analytical_grad, numerical_grad,
                              "SH means3D (degree " + std::to_string(sh_degree) + ")",
                              5e-3, 5e-3);  // Slightly looser tolerance
        }
    }
}

TEST_F(NumericalGradientTest, ProjectionGradientTest) {
    torch::manual_seed(42);

    int N = 5;
    int C = 1;
    int width = 64, height = 64;

    auto means = torch::randn({N, 3}, device);
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

    auto K = torch::tensor({
                               {50.0f, 0.0f, 32.0f},
                               {0.0f, 50.0f, 32.0f},
                               {0.0f, 0.0f, 1.0f}
                           }, device).unsqueeze(0);
    auto settings = torch::tensor({(float)width, (float)height, 0.3f, 0.01f, 1000.0f, 0.0f, 1.0f}, device);

    // Test means gradient
    {
        auto means_func = [&](torch::Tensor x) -> torch::Tensor {
            torch::NoGradGuard no_grad;
            auto empty_covars = torch::empty({0, 3, 3}, x.options());
            auto [radii, means2d, depths, conics, compensations] =
                gsplat::projection_ewa_3dgs_fused_fwd(
                    x, empty_covars, quats.detach(), scales.detach(),
                    opacities.detach(), viewmat.detach(), K,
                    width, height, 0.3f, 0.01f, 1000.0f, 0.0f, false,
                    gsplat::CameraModelType::PINHOLE);
            // Only consider visible Gaussians
            auto valid = (radii > 0).all(-1);
            return (means2d * valid.unsqueeze(-1).to(torch::kFloat32)).sum();
        };

        auto outputs = ProjectionFunction::apply(
            means, quats, scales, opacities, viewmat, K, settings);

        auto radii = outputs[0];
        auto means2d = outputs[1];
        auto valid = (radii > 0).all(-1);
        auto loss = (means2d * valid.unsqueeze(-1).to(torch::kFloat32)).sum();
        loss.backward();
        auto analytical_grad = means.grad().clone();

        auto numerical_grad = compute_numerical_gradient(means_func, means.detach(), eps);

        compare_gradients(analytical_grad, numerical_grad, "projection means");
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

        auto loss = covars.sum() + precis.sum() * 1e-6f;
        loss.backward();

        auto quats_grad_finite = torch::isfinite(quats.grad()).all().item<bool>();
        auto scales_grad_finite = torch::isfinite(scales.grad()).all().item<bool>();

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

        auto loss = covars.sum();
        loss.backward();

        auto quats_grad_finite = torch::isfinite(quats.grad()).all().item<bool>();

        EXPECT_TRUE(quats_grad_finite)
            << "Non-finite gradients with near-zero quaternions";
    }
}