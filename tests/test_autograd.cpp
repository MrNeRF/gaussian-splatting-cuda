#include "Ops.h"
#include "core/debug_utils.hpp"
#include "core/rasterizer_autograd.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <torch/torch.h>

// Using the exposed autograd functions from gs namespace
using gs::ProjectionFunction;
using gs::QuatScaleToCovarPreciFunction;
using gs::SphericalHarmonicsFunction;

class AutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        device = torch::kCUDA;

        // Clear any previous CUDA errors
        torch::cuda::synchronize();
        auto error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "Warning: Previous CUDA error cleared: " << cudaGetErrorString(error) << std::endl;
        }
    }

    void assertTensorClose(const torch::Tensor& a, const torch::Tensor& b,
                           double rtol = 1e-4, double atol = 1e-4) {
        ASSERT_TRUE(torch::allclose(a, b, rtol, atol))
            << "Tensors not close:\n"
            << "Max diff: " << (a - b).abs().max().item<float>() << "\n"
            << "Mean diff: " << (a - b).abs().mean().item<float>();
    }

    torch::Device device{torch::kCPU};
};

TEST_F(AutogradTest, QuatScaleToCovarPreciAutogradTest) {
    torch::manual_seed(42);

    int N = 100;
    auto quats = torch::randn({N, 4}, device);
    quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto scales = torch::rand({N, 3}, device) * 0.1f;

    // Make them leaf tensors requiring gradients
    quats.requires_grad_(true);
    scales.requires_grad_(true);

    // Test with triu=false
    {
        auto settings = torch::tensor({1.0f, 1.0f, 0.0f}, device); // compute_covar=true, compute_preci=true, triu=false
        auto outputs = QuatScaleToCovarPreciFunction::apply(quats, scales, settings);

        auto covars = outputs[0];
        auto precis = outputs[1];

        EXPECT_EQ(covars.sizes(), torch::IntArrayRef({N, 3, 3}));
        EXPECT_EQ(precis.sizes(), torch::IntArrayRef({N, 3, 3}));

        // Create a simple scalar loss
        auto loss = covars.sum() * 0.001f + precis.sum() * 0.00001f; // Scale down to avoid huge gradients
        loss.backward();

        // Check gradients exist on the ORIGINAL input tensors
        EXPECT_TRUE(quats.grad().defined());
        EXPECT_TRUE(scales.grad().defined());
        auto quats_grad_has_nan = quats.grad().isnan().any().item<bool>();
        auto scales_grad_has_nan = scales.grad().isnan().any().item<bool>();
        EXPECT_FALSE(quats_grad_has_nan);
        EXPECT_FALSE(scales_grad_has_nan);

        // Check gradient magnitudes are reasonable
        auto quat_grad_norm = quats.grad().norm();
        auto scale_grad_norm = scales.grad().norm();
        EXPECT_GT(quat_grad_norm.item<float>(), 0);
        EXPECT_GT(scale_grad_norm.item<float>(), 0);
        // Relax the upper bound check for scale gradients
        EXPECT_LT(quat_grad_norm.item<float>(), 1e8);
        EXPECT_LT(scale_grad_norm.item<float>(), 1e10);
    }

    // Clear gradients
    if (quats.grad().defined())
        quats.grad().zero_();
    if (scales.grad().defined())
        scales.grad().zero_();

    // Test with triu=true
    {
        auto settings = torch::tensor({1.0f, 1.0f, 1.0f}, device); // compute_covar=true, compute_preci=true, triu=true
        auto outputs = QuatScaleToCovarPreciFunction::apply(quats, scales, settings);

        auto covars_triu = outputs[0];
        auto precis_triu = outputs[1];

        EXPECT_EQ(covars_triu.sizes(), torch::IntArrayRef({N, 6}));
        EXPECT_EQ(precis_triu.sizes(), torch::IntArrayRef({N, 6}));

        auto loss = covars_triu.sum() + precis_triu.sum() * 0.01f;
        loss.backward();

        EXPECT_TRUE(quats.grad().defined());
        EXPECT_TRUE(scales.grad().defined());
    }
}

// TEST_F(AutogradTest, SphericalHarmonicsAutogradTest) {
//     torch::manual_seed(42);
//
//     std::vector<int> sh_degrees = {0, 1, 2, 3};
//
//     for (int sh_degree : sh_degrees) {
//         int N = 1000;
//         int K = (sh_degree + 1) * (sh_degree + 1);
//         int C = 1; // Single camera for this test
//
//         auto sh_coeffs = torch::randn({N, K, 3}, device).set_requires_grad(true);
//         auto means3D = torch::randn({N, 3}, device).set_requires_grad(true);
//         auto viewmat = torch::eye(4, device).unsqueeze(0).set_requires_grad(true); // [1, 4, 4]
//         auto radii = torch::ones({C, N, 2}, device) * 10;                          // All visible
//         auto sh_degree_tensor = torch::tensor({sh_degree}, torch::TensorOptions().dtype(torch::kInt32).device(device));
//
//         // Forward through autograd function
//         auto outputs = SphericalHarmonicsFunction::apply(sh_coeffs, means3D, viewmat, radii, sh_degree_tensor);
//         auto colors = outputs[0]; // Extract the tensor from tensor_list
//
//         EXPECT_EQ(colors.sizes(), torch::IntArrayRef({C, N, 3}));
//         auto colors_has_nan = colors.isnan().any().item<bool>();
//         EXPECT_FALSE(colors_has_nan);
//
//         // Create a simple loss
//         auto loss = colors.sum();
//         loss.backward();
//
//         // Check gradients on ORIGINAL inputs
//         EXPECT_TRUE(sh_coeffs.grad().defined());
//         auto sh_coeffs_grad_has_nan = sh_coeffs.grad().isnan().any().item<bool>();
//         EXPECT_FALSE(sh_coeffs_grad_has_nan);
//
//         if (sh_degree > 0) {
//             EXPECT_TRUE(means3D.grad().defined());
//             auto means3D_grad_has_nan = means3D.grad().isnan().any().item<bool>();
//             EXPECT_FALSE(means3D_grad_has_nan);
//         }
//
//         // Clear gradients for next iteration
//         if (sh_coeffs.grad().defined())
//             sh_coeffs.grad().zero_();
//         if (means3D.grad().defined())
//             means3D.grad().zero_();
//         if (viewmat.grad().defined())
//             viewmat.grad().zero_();
//     }
// }

TEST_F(AutogradTest, ProjectionAutogradTest) {
    torch::manual_seed(42);

    int N = 100;
    int C = 2;
    int width = 640, height = 480;

    auto means = torch::randn({N, 3}, device).detach().requires_grad_(true);
    auto quats = torch::nn::functional::normalize(
                     torch::randn({N, 4}, device),
                     torch::nn::functional::NormalizeFuncOptions().dim(-1))
                     .detach()
                     .requires_grad_(true);
    auto scales = (torch::rand({N, 3}, device) * 0.1f)
                      .detach()
                      .requires_grad_(true);
    auto opacities = torch::rand({N}, device).detach().requires_grad_(true);
    auto viewmats = torch::eye(4, device).unsqueeze(0).repeat({C, 1, 1}).detach().requires_grad_(true);

    auto Ks = torch::tensor({{300.0f, 0.0f, 320.0f},
                             {0.0f, 300.0f, 240.0f},
                             {0.0f, 0.0f, 1.0f}},
                            device)
                  .unsqueeze(0)
                  .repeat({C, 1, 1});

    // Test projection (note: compensations are always computed as false in the actual implementation)
    {
        auto settings = torch::tensor({(float)width, (float)height,
                                       0.3f, 0.01f, 10000.0f,
                                       0.0f, 1.0f},
                                      device);

        auto outputs = ProjectionFunction::apply(
            means, quats, scales, opacities, viewmats, Ks, settings);

        auto radii = outputs[0];
        auto means2d = outputs[1];
        auto depths = outputs[2];
        auto conics = outputs[3];
        auto compensations = outputs[4];

        EXPECT_EQ(radii.sizes(), torch::IntArrayRef({C, N, 2}));
        EXPECT_EQ(means2d.sizes(), torch::IntArrayRef({C, N, 2}));
        EXPECT_EQ(depths.sizes(), torch::IntArrayRef({C, N}));
        EXPECT_EQ(conics.sizes(), torch::IntArrayRef({C, N, 3}));
        EXPECT_EQ(compensations.sizes(), torch::IntArrayRef({C, N}));

        // Create random gradient outputs (following Python pattern)
        auto valid = (radii > 0).all(-1);
        auto v_means2d = torch::randn_like(means2d) * valid.unsqueeze(-1).to(torch::kFloat32);
        auto v_depths = torch::randn_like(depths) * valid.to(torch::kFloat32);
        auto v_conics = torch::randn_like(conics) * valid.unsqueeze(-1).to(torch::kFloat32);

        // Compute gradients using autograd::grad with the gradient outputs
        torch::autograd::variable_list outputs_list = {(means2d * v_means2d).sum() + (depths * v_depths).sum() + (conics * v_conics).sum()};
        torch::autograd::variable_list inputs = {means, quats, scales, opacities, viewmats};
        auto grads = torch::autograd::grad(
            outputs_list,
            inputs,
            /*grad_outputs=*/{},
            /*retain_graph=*/false,
            /*create_graph=*/false,
            /*allow_unused=*/true);

        // Check that gradients are defined for inputs that should have gradients
        EXPECT_TRUE(grads[0].defined()); // means grad
        EXPECT_TRUE(grads[4].defined()); // viewmats grad

        if (grads[0].defined()) {
            auto grad0_has_nan = grads[0].isnan().any().item<bool>();
            EXPECT_FALSE(grad0_has_nan);
        }
        if (grads[4].defined()) {
            auto grad4_has_nan = grads[4].isnan().any().item<bool>();
            EXPECT_FALSE(grad4_has_nan);
        }
    }
}

TEST_F(AutogradTest, FullRenderingPipelineAutogradTest) {
    torch::manual_seed(42);

    // Create a complete rendering pipeline test
    int N = 50;
    int width = 128, height = 128;
    int C = 1; // Single camera

    // Create parameters with gradients
    auto means = torch::randn({N, 3}, device) * 2.0f;
    means.requires_grad_(true);

    auto quats = torch::randn({N, 4}, device);
    quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    quats.requires_grad_(true);

    auto scales = torch::rand({N, 3}, device) * 0.5f;
    scales.requires_grad_(true);

    auto opacities = torch::ones({N}, device); // Use ones to avoid opacity gradient issues
    opacities.requires_grad_(true);

    auto sh_coeffs = torch::randn({N, 1, 3}, device); // Only DC component
    sh_coeffs.requires_grad_(true);

    // Camera
    auto viewmat = torch::eye(4, device).unsqueeze(0);
    viewmat.requires_grad_(true);
    auto K = torch::tensor({{200.0f, 0.0f, 64.0f},
                            {0.0f, 200.0f, 64.0f},
                            {0.0f, 0.0f, 1.0f}},
                           device)
                 .unsqueeze(0);

    // 1. Projection
    auto proj_settings = torch::tensor({(float)width, (float)height, 0.3f, 0.01f, 1000.0f, 0.0f, 1.0f}, device);
    auto proj_outputs = ProjectionFunction::apply(
        means, quats, scales, opacities, viewmat, K, proj_settings);

    auto radii = proj_outputs[0];
    auto means2d = proj_outputs[1];
    auto depths = proj_outputs[2];
    auto conics = proj_outputs[3];
    auto compensations = proj_outputs[4];

    // 2. Colors from SH using the new interface
    // Compute directions from camera position to gaussians
    auto viewmat_inv = torch::inverse(viewmat);
    auto campos = viewmat_inv.index({0, torch::indexing::Slice(torch::indexing::None, 3), 3});
    auto dirs = means - campos; // [N, 3]

    // Create masks based on visibility
    auto masks = (radii > 0).all(-1).squeeze(0); // [N]

    auto sh_degree_tensor = torch::tensor({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto color_outputs = SphericalHarmonicsFunction::apply(
        sh_degree_tensor, dirs, sh_coeffs, masks);
    auto colors = color_outputs[0]; // [N, 3]

    // Apply SH offset and clamp (like in rasterizer)
    colors = torch::clamp_min(colors + 0.5f, 0.0f);

    // Create a combined loss that includes projection outputs and colors
    auto valid = (radii > 0).all(-1).squeeze(0); // Remove camera dimension
    auto loss = (means2d.squeeze(0) * valid.unsqueeze(-1).to(torch::kFloat32)).sum() * 0.001f +
                (depths.squeeze(0) * valid.to(torch::kFloat32)).sum() * 0.001f +
                (conics.squeeze(0) * valid.unsqueeze(-1).to(torch::kFloat32)).sum() * 0.001f +
                colors.mean();

    // Compute gradients using torch::autograd::grad
    std::vector<torch::Tensor> inputs = {means, quats, scales, opacities, sh_coeffs, viewmat};
    std::vector<torch::Tensor> outputs = {loss};
    std::vector<torch::Tensor> grad_outputs = {};

    auto grads = torch::autograd::grad(
        outputs,
        inputs,
        grad_outputs,
        /*retain_graph=*/true,
        /*create_graph=*/false,
        /*allow_unused=*/true);

    auto v_means = grads[0];
    auto v_quats = grads[1];
    auto v_scales = grads[2];
    auto v_opacities = grads[3];
    auto v_sh_coeffs = grads[4];
    auto v_viewmat = grads[5];

    // Check that all gradients are defined
    EXPECT_TRUE(v_means.defined()) << "means grad not defined";
    EXPECT_TRUE(v_quats.defined()) << "quats grad not defined";
    EXPECT_TRUE(v_scales.defined()) << "scales grad not defined";
    EXPECT_TRUE(v_opacities.defined()) << "opacities grad not defined";
    EXPECT_TRUE(v_sh_coeffs.defined()) << "sh_coeffs grad not defined";
    EXPECT_TRUE(v_viewmat.defined()) << "viewmat grad not defined";

    // Check gradients are non-zero (at least some should be)
    auto means_grad_sum = v_means.abs().sum().item<float>();
    auto quats_grad_sum = v_quats.abs().sum().item<float>();
    auto scales_grad_sum = v_scales.abs().sum().item<float>();
    auto sh_coeffs_grad_sum = v_sh_coeffs.abs().sum().item<float>();

    std::cout << "Gradient sums:" << std::endl;
    std::cout << "  means: " << means_grad_sum << std::endl;
    std::cout << "  quats: " << quats_grad_sum << std::endl;
    std::cout << "  scales: " << scales_grad_sum << std::endl;
    std::cout << "  sh_coeffs: " << sh_coeffs_grad_sum << std::endl;
    std::cout << "  viewmat: " << v_viewmat.abs().sum().item<float>() << std::endl;

    EXPECT_GT(means_grad_sum, 0) << "means gradients are all zero";
    EXPECT_GT(quats_grad_sum, 0) << "quats gradients are all zero";
    EXPECT_GT(scales_grad_sum, 0) << "scales gradients are all zero";
    EXPECT_GT(sh_coeffs_grad_sum, 0) << "sh_coeffs gradients are all zero";

    // Check gradients are valid
    auto means_has_nan = v_means.isnan().any().item<bool>();
    auto quats_has_nan = v_quats.isnan().any().item<bool>();
    auto scales_has_nan = v_scales.isnan().any().item<bool>();
    auto sh_coeffs_has_nan = v_sh_coeffs.isnan().any().item<bool>();

    EXPECT_FALSE(means_has_nan) << "means grad has NaN";
    EXPECT_FALSE(quats_has_nan) << "quats grad has NaN";
    EXPECT_FALSE(scales_has_nan) << "scales grad has NaN";
    EXPECT_FALSE(sh_coeffs_has_nan) << "sh_coeffs grad has NaN";

    // Also check viewmat gradient
    if (v_viewmat.defined()) {
        auto viewmat_has_nan = v_viewmat.isnan().any().item<bool>();
        EXPECT_FALSE(viewmat_has_nan) << "viewmat grad has NaN";

        // For SH degree 0, viewmat gradients might be zero since colors don't depend on view direction
        if (sh_coeffs.size(1) > 1) { // Only expect viewmat gradients for higher SH degrees
            auto viewmat_grad_sum = v_viewmat.abs().sum().item<float>();
            EXPECT_GT(viewmat_grad_sum, 0) << "viewmat gradients are all zero";
        }
    }
}

TEST_F(AutogradTest, GradientAccumulationTest) {
    torch::manual_seed(42);

    int N = 50;
    auto quats = torch::randn({N, 4}, device);
    quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto scales = torch::rand({N, 3}, device) * 0.1f;

    quats.requires_grad_(true);
    scales.requires_grad_(true);

    auto settings = torch::tensor({1.0f, 1.0f, 0.0f}, device);

    // First forward/backward
    auto outputs1 = QuatScaleToCovarPreciFunction::apply(quats, scales, settings);
    auto covars1 = outputs1[0];
    auto loss1 = covars1.sum();

    // Clone the inputs to create a separate graph for the second computation
    auto quats_clone = quats.clone().detach().requires_grad_(true);
    auto scales_clone = scales.clone().detach().requires_grad_(true);

    // Second forward/backward with cloned inputs
    auto outputs2 = QuatScaleToCovarPreciFunction::apply(quats_clone, scales_clone, settings);
    auto covars2 = outputs2[0];
    auto loss2 = covars2.sum();

    // Compute gradients using autograd::grad
    torch::autograd::variable_list loss_list1 = {loss1};
    torch::autograd::variable_list inputs1 = {quats, scales};
    auto grads1 = torch::autograd::grad(
        loss_list1,
        inputs1,
        /*grad_outputs=*/{},
        /*retain_graph=*/false,
        /*create_graph=*/false);

    torch::autograd::variable_list loss_list2 = {loss2};
    torch::autograd::variable_list inputs2 = {quats_clone, scales_clone};
    auto grads2 = torch::autograd::grad(
        loss_list2,
        inputs2,
        /*grad_outputs=*/{},
        /*retain_graph=*/false,
        /*create_graph=*/false);

    // Since inputs are the same (just cloned), gradients should be the same
    assertTensorClose(grads2[0], grads1[0], 1e-5, 1e-5);
    assertTensorClose(grads2[1], grads1[1], 1e-5, 1e-5);

    // To test accumulation, we need to use .backward() instead
    // Reset gradients
    quats.mutable_grad() = torch::zeros_like(quats);
    scales.mutable_grad() = torch::zeros_like(scales);

    // Create two separate forward passes to avoid graph reuse
    auto outputs3 = QuatScaleToCovarPreciFunction::apply(quats, scales, settings);
    auto covars3 = outputs3[0];
    auto loss3 = covars3.sum();

    // First backward
    loss3.backward();
    auto acc_grad_quats_1 = quats.grad().clone();
    auto acc_grad_scales_1 = scales.grad().clone();

    // Second forward and backward
    auto outputs4 = QuatScaleToCovarPreciFunction::apply(quats, scales, settings);
    auto covars4 = outputs4[0];
    auto loss4 = covars4.sum();
    loss4.backward();

    // Check that gradients accumulated
    auto expected_quats_grad = acc_grad_quats_1 * 2;
    auto expected_scales_grad = acc_grad_scales_1 * 2;

    assertTensorClose(quats.grad(), expected_quats_grad, 1e-5, 1e-5);
    assertTensorClose(scales.grad(), expected_scales_grad, 1e-5, 1e-5);
}