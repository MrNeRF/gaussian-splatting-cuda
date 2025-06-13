#include "Ops.h"
#include "core/rasterizer_autograd.hpp"
#include "test_data_loader.hpp"
#include "torch_impl.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <torch/torch.h>

using namespace test_utils;
using gs::ProjectionFunction;
using gs::QuatScaleToCovarPreciFunction;
using gs::RasterizationFunction;
using gs::SphericalHarmonicsFunction;

class BasicPyTorchDataTest : public ::testing::Test {
protected:
    static TestData test_data;
    static torch::Device device;
    static bool data_loaded;

    static void SetUpTestSuite() {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }

        device = torch::kCUDA;

        if (!data_loaded) {
            try {
                // Load test data from .pt file
                test_data = load_test_data(device);
                data_loaded = true;

                std::cout << "\nTest data loaded successfully:" << std::endl;
                std::cout << "  means: " << test_data.means.sizes() << std::endl;
                std::cout << "  quats: " << test_data.quats.sizes() << std::endl;
                std::cout << "  scales: " << test_data.scales.sizes() << std::endl;
                std::cout << "  opacities: " << test_data.opacities.sizes() << std::endl;
                std::cout << "  colors: " << test_data.colors.sizes() << std::endl;
                std::cout << "  viewmats: " << test_data.viewmats.sizes() << std::endl;
                std::cout << "  Ks: " << test_data.Ks.sizes() << std::endl;
                std::cout << "  Resolution: " << test_data.width << "x" << test_data.height << "\n"
                          << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to load test data: " << e.what() << std::endl;
                GTEST_SKIP() << "Test data not available";
            }
        }

        // Clear any previous CUDA errors
        torch::cuda::synchronize();
        cudaGetLastError();
    }

    void assertTensorClose(const torch::Tensor& a, const torch::Tensor& b,
                           double rtol = 1e-4, double atol = 1e-4,
                           const std::string& name = "") {
        ASSERT_TRUE(torch::allclose(a, b, rtol, atol))
            << "Tensors not close" << (name.empty() ? "" : " for " + name) << ":\n"
            << "  Shape A: " << a.sizes() << ", Shape B: " << b.sizes() << "\n"
            << "  Max diff: " << (a - b).abs().max().item<float>() << "\n"
            << "  Mean diff: " << (a - b).abs().mean().item<float>() << "\n"
            << "  A sample: " << a.view(-1).slice(0, 0, 5) << "\n"
            << "  B sample: " << b.view(-1).slice(0, 0, 5);
    }
};

// Static member definitions
TestData BasicPyTorchDataTest::test_data;
torch::Device BasicPyTorchDataTest::device{torch::kCPU};
bool BasicPyTorchDataTest::data_loaded = false;

// Test data loading
TEST_F(BasicPyTorchDataTest, DataLoadingTest) {
    // This test verifies that data was loaded correctly
    EXPECT_GT(test_data.means.size(0), 0) << "No Gaussians loaded";
    EXPECT_GT(test_data.viewmats.size(0), 0) << "No cameras loaded";
    EXPECT_GT(test_data.width, 0) << "Invalid width";
    EXPECT_GT(test_data.height, 0) << "Invalid height";

    // Check data types
    EXPECT_EQ(test_data.means.dtype(), torch::kFloat32);
    EXPECT_EQ(test_data.quats.dtype(), torch::kFloat32);
    EXPECT_EQ(test_data.scales.dtype(), torch::kFloat32);
    EXPECT_EQ(test_data.opacities.dtype(), torch::kFloat32);
    EXPECT_EQ(test_data.colors.dtype(), torch::kFloat32);
    EXPECT_EQ(test_data.viewmats.dtype(), torch::kFloat32);
    EXPECT_EQ(test_data.Ks.dtype(), torch::kFloat32);

    // Check dimensions
    const int N = test_data.means.size(0);
    const int C = test_data.viewmats.size(0);

    EXPECT_EQ(test_data.means.sizes(), torch::IntArrayRef({N, 3}));
    EXPECT_EQ(test_data.quats.sizes(), torch::IntArrayRef({N, 4}));
    EXPECT_EQ(test_data.scales.sizes(), torch::IntArrayRef({N, 3}));
    EXPECT_EQ(test_data.opacities.sizes(), torch::IntArrayRef({N}));
    EXPECT_EQ(test_data.viewmats.sizes(), torch::IntArrayRef({C, 4, 4}));
    EXPECT_EQ(test_data.Ks.sizes(), torch::IntArrayRef({C, 3, 3}));

    // Colors can be either [N, 3] or [C, N, 3]
    if (test_data.colors.dim() == 2) {
        EXPECT_EQ(test_data.colors.sizes(), torch::IntArrayRef({N, 3}));
    } else {
        EXPECT_EQ(test_data.colors.sizes(), torch::IntArrayRef({C, N, 3}));
    }

    // Check value ranges
    EXPECT_TRUE((test_data.opacities >= 0).all().item<bool>())
        << "Negative opacities found";
    EXPECT_TRUE((test_data.opacities <= 1).all().item<bool>())
        << "Opacities > 1 found";

    EXPECT_TRUE((test_data.colors >= 0).all().item<bool>())
        << "Negative colors found";
    EXPECT_TRUE((test_data.colors <= 1).all().item<bool>())
        << "Colors > 1 found";

    // Check quaternions are normalized
    auto quat_norms = test_data.quats.norm(2, -1);
    EXPECT_TRUE(torch::allclose(quat_norms, torch::ones_like(quat_norms), 1e-5, 1e-5))
        << "Quaternions are not normalized";

    // Check no NaN values
    EXPECT_FALSE(test_data.means.isnan().any().item<bool>());
    EXPECT_FALSE(test_data.quats.isnan().any().item<bool>());
    EXPECT_FALSE(test_data.scales.isnan().any().item<bool>());
    EXPECT_FALSE(test_data.opacities.isnan().any().item<bool>());
    EXPECT_FALSE(test_data.colors.isnan().any().item<bool>());
    EXPECT_FALSE(test_data.viewmats.isnan().any().item<bool>());
    EXPECT_FALSE(test_data.Ks.isnan().any().item<bool>());
}

// Test quat_scale_to_covar_preci matching Python test
TEST_F(BasicPyTorchDataTest, QuatScaleToCovarPreciTest) {
    torch::manual_seed(42);

    auto quats = test_data.quats.clone().set_requires_grad(true);
    auto scales = test_data.scales.clone().set_requires_grad(true);

    // Test with triu=false
    {
        // Forward - gsplat implementation
        auto [covars, precis] = gsplat::quat_scale_to_covar_preci_fwd(
            quats, scales, true, true, false);

        // Forward - reference implementation
        auto [_covars, _precis] = reference::quat_scale_to_covar_preci(
            quats, scales, true, true, false);

        assertTensorClose(covars, _covars, 1e-5, 1e-5, "covars");
        // Note: Precision matrices can have larger numerical differences
        assertTensorClose(precis, _precis, 1e-1, 1e-1, "precis");

        // Backward test using autograd
        auto settings = torch::tensor({1.0f, 1.0f, 0.0f}, device);
        auto outputs = QuatScaleToCovarPreciFunction::apply(quats, scales, settings);
        covars = outputs[0];
        precis = outputs[1];

        auto v_covars = torch::randn_like(covars);
        auto v_precis = torch::randn_like(precis) * 0.01f;

        auto loss = (covars * v_covars + precis * v_precis).sum();

        // Compute gradients
        torch::autograd::variable_list outputs_list = {loss};
        torch::autograd::variable_list inputs_list = {quats, scales};
        auto grads = torch::autograd::grad(outputs_list, inputs_list, {}, true);
        auto v_quats = grads[0];
        auto v_scales = grads[1];

        // Reference gradients
        auto ref_loss = (_covars * v_covars + _precis * v_precis).sum();
        torch::autograd::variable_list ref_outputs = {ref_loss};
        auto ref_grads = torch::autograd::grad(ref_outputs, inputs_list, {}, true);
        auto _v_quats = ref_grads[0];
        auto _v_scales = ref_grads[1];

        assertTensorClose(v_quats, _v_quats, 1e-1, 1e-1, "grad_quats");
        assertTensorClose(v_scales, _v_scales, 1e-1, 1e-1, "grad_scales");
    }

    // Test with triu=true
    {
        auto [covars, precis] = gsplat::quat_scale_to_covar_preci_fwd(
            quats, scales, true, true, true);
        auto [_covars, _precis] = reference::quat_scale_to_covar_preci(
            quats, scales, true, true, true);

        EXPECT_EQ(covars.sizes(), torch::IntArrayRef({quats.size(0), 6}));
        EXPECT_EQ(precis.sizes(), torch::IntArrayRef({quats.size(0), 6}));

        assertTensorClose(covars, _covars, 1e-5, 1e-5, "covars_triu");
        assertTensorClose(precis, _precis, 1e-1, 1e-1, "precis_triu");
    }
}

// Test projection matching Python test
TEST_F(BasicPyTorchDataTest, ProjectionTest) {
    torch::manual_seed(42);

    auto means = test_data.means.clone().set_requires_grad(true);
    auto quats = test_data.quats.clone().set_requires_grad(true);
    auto scales = test_data.scales.clone().set_requires_grad(true);
    auto viewmats = test_data.viewmats.clone().set_requires_grad(true);
    auto Ks = test_data.Ks;

    const int width = test_data.width;
    const int height = test_data.height;
    const float eps2d = 0.3f;
    const float near_plane = 0.01f;
    const float far_plane = 10000.0f;
    const bool calc_compensations = false;

    // Fused projection test
    {
        // Prepare settings tensor for ProjectionFunction
        auto proj_settings = torch::tensor({(float)width, (float)height,
                                            eps2d, near_plane, far_plane,
                                            0.0f, 1.0f}, // radius_clip=0.0, scaling_modifier=1.0
                                           device);

        // Create opacities tensor of all ones
        const int N = means.size(0);
        auto opacities = torch::ones({N}, means.options());

        // gsplat implementation using ProjectionFunction
        auto outputs = ProjectionFunction::apply(
            means, quats, scales, opacities,
            viewmats, Ks, proj_settings);

        auto radii = outputs[0];
        auto means2d = outputs[1];
        auto depths = outputs[2];
        auto conics = outputs[3];
        auto compensations = outputs[4];

        // Reference implementation
        auto [_covars, _] = reference::quat_scale_to_covar_preci(quats, scales, true, false, false);
        auto [_radii, _means2d, _depths, _conics, _compensations] =
            reference::fully_fused_projection(
                means, _covars, viewmats, Ks, width, height,
                eps2d, near_plane, far_plane, calc_compensations, "pinhole");

        // Compare (radii is integer so allow 1 unit difference)
        auto valid = (radii > 0).all(-1) & (_radii > 0).all(-1);

        EXPECT_LE((radii - _radii).abs().max().item<int>(), 1)
            << "Radii differ by more than 1";

        // Only compare valid Gaussians
        if (valid.any().item<bool>()) {
            // For each camera, compare valid projections
            for (int c = 0; c < viewmats.size(0); ++c) {
                auto cam_valid = valid[c];
                if (cam_valid.any().item<bool>()) {
                    auto idx = cam_valid.nonzero().squeeze(-1);

                    assertTensorClose(
                        means2d[c].index_select(0, idx),
                        _means2d[c].index_select(0, idx),
                        1e-4, 1e-4, "means2d_cam" + std::to_string(c));

                    assertTensorClose(
                        depths[c].index_select(0, idx),
                        _depths[c].index_select(0, idx),
                        1e-4, 1e-4, "depths_cam" + std::to_string(c));

                    assertTensorClose(
                        conics[c].index_select(0, idx),
                        _conics[c].index_select(0, idx),
                        1e-4, 1e-4, "conics_cam" + std::to_string(c));
                }
            }
        }
    }

    // Backward test using torch::autograd::grad (identical to Python test)
    {
        // Clone tensors and set requires_grad to match Python test
        means = test_data.means.clone().set_requires_grad(true);
        quats = test_data.quats.clone().set_requires_grad(true);
        scales = test_data.scales.clone().set_requires_grad(true);
        viewmats = test_data.viewmats.clone().set_requires_grad(true);
        auto opacities = test_data.opacities.clone().set_requires_grad(true);

        // Prepare settings tensor for ProjectionFunction
        auto proj_settings = torch::tensor({(float)width, (float)height,
                                            eps2d, near_plane, far_plane,
                                            0.0f, 1.0f}, // radius_clip=0.0, scaling_modifier=1.0
                                           device);

        // Forward pass - using ProjectionFunction (equivalent to fully_fused_projection with fused=true)
        auto outputs = ProjectionFunction::apply(
            means, quats, scales, opacities, viewmats, Ks, proj_settings);

        auto radii = outputs[0];
        auto means2d = outputs[1];
        auto depths = outputs[2];
        auto conics = outputs[3];
        auto compensations = outputs[4];

        // Reference forward - using non-triu covariance for reference
        auto [_covars, _] = reference::quat_scale_to_covar_preci(quats, scales, true, false, false);
        auto [_radii, _means2d, _depths, _conics, _compensations] =
            reference::fully_fused_projection(
                means, _covars, viewmats, Ks, width, height,
                eps2d, near_plane, far_plane, calc_compensations, "pinhole");

        // Create valid mask (matching Python: valid = (radii > 0).all(dim=-1) & (_radii > 0).all(dim=-1))
        auto valid = (radii > 0).all(-1) & (_radii > 0).all(-1);

        // Create random gradients for valid Gaussians (matching Python exactly)
        auto v_means2d = torch::randn_like(means2d) * valid.unsqueeze(-1).to(torch::kFloat32);
        auto v_depths = torch::randn_like(depths) * valid.to(torch::kFloat32);
        auto v_conics = torch::randn_like(conics) * valid.unsqueeze(-1).to(torch::kFloat32);
        torch::Tensor v_compensations;
        if (calc_compensations && compensations.defined()) {
            v_compensations = torch::randn_like(compensations) * valid.to(torch::kFloat32);
        }

        // Compute loss for our implementation (matching Python exactly)
        auto loss = (means2d * v_means2d).sum() +
                    (depths * v_depths).sum() +
                    (conics * v_conics).sum();
        if (calc_compensations && v_compensations.defined()) {
            loss = loss + (compensations * v_compensations).sum();
        }

        // Compute gradients using torch::autograd::grad (matching Python order: viewmats, quats, scales, means)
        std::vector<torch::Tensor> inputs = {viewmats, quats, scales, means};
        auto grads = torch::autograd::grad(
            {loss},
            inputs,
            {},
            /*retain_graph=*/false,
            /*create_graph=*/false,
            /*allow_unused=*/false);

        auto v_viewmats = grads[0];
        auto v_quats = grads[1];
        auto v_scales = grads[2];
        auto v_means = grads[3];

        // Compute loss for reference implementation
        auto _loss = (_means2d * v_means2d).sum() +
                     (_depths * v_depths).sum() +
                     (_conics * v_conics).sum();
        if (calc_compensations && v_compensations.defined()) {
            _loss = _loss + (_compensations * v_compensations).sum();
        }

        // Compute reference gradients
        auto _grads = torch::autograd::grad(
            {_loss},
            inputs,
            {},
            /*retain_graph=*/false,
            /*create_graph=*/false,
            /*allow_unused=*/false);

        auto _v_viewmats = _grads[0];
        auto _v_quats = _grads[1];
        auto _v_scales = _grads[2];
        auto _v_means = _grads[3];

        // Compare gradients with reference implementation using Python test tolerances
        assertTensorClose(v_viewmats, _v_viewmats, 2e-3, 2e-3, "viewmats gradients");
        assertTensorClose(v_quats, _v_quats, 2e-1, 2e-2, "quats gradients");
        assertTensorClose(v_scales, _v_scales, 1e-1, 2e-1, "scales gradients");
        assertTensorClose(v_means, _v_means, 1e-2, 6e-2, "means gradients");
    }
}

// Test spherical harmonics gradient flow
TEST_F(BasicPyTorchDataTest, SphericalHarmonicsTest) {
    torch::manual_seed(42);

    std::vector<int> sh_degrees = {0, 1, 2, 3};

    for (int sh_degree : sh_degrees) {
        std::cout << "Testing SH degree " << sh_degree << std::endl;

        // Use a small subset for testing
        const int N = 100;
        const int K = (3 + 1) * (3 + 1); // Max degree 3
        const int C = 1;

        // Use test data to create realistic directions
        auto means3D = test_data.means.slice(0, 0, N).clone();
        auto viewmat = test_data.viewmats[0].unsqueeze(0).clone();
        auto coeffs = torch::randn({N, K, 3}, device);

        // Compute camera position from inverse viewmat
        auto viewmat_inv = torch::inverse(viewmat);
        auto campos = viewmat_inv.index({0, torch::indexing::Slice(torch::indexing::None, 3), 3});

        // Compute directions from camera to each Gaussian
        auto dirs = means3D - campos; // [N, 3]
        dirs = dirs.detach();         // Detach to test SH function in isolation

        // Create masks (all visible)
        auto masks = torch::ones({N}, torch::TensorOptions().dtype(torch::kBool).device(device));

        // Set requires_grad for the actual SH inputs
        dirs.requires_grad_(true);
        coeffs.requires_grad_(true);

        // Create sh_degree tensor
        auto sh_degree_tensor = torch::tensor({sh_degree},
                                              torch::TensorOptions().dtype(torch::kInt32).device(device));

        // CUDA implementation using SphericalHarmonicsFunction
        auto colors = SphericalHarmonicsFunction::apply(
            sh_degree_tensor, dirs, coeffs, masks)[0];

        // Reference implementation
        auto _colors = reference::spherical_harmonics(sh_degree, dirs, coeffs);

        // Basic checks
        EXPECT_EQ(colors.sizes(), torch::IntArrayRef({N, 3}));
        EXPECT_FALSE(colors.isnan().any().item<bool>())
            << "NaN in colors for degree " << sh_degree;
        EXPECT_FALSE(colors.isinf().any().item<bool>())
            << "Inf in colors for degree " << sh_degree;

        // Forward should match reference
        EXPECT_TRUE(torch::allclose(colors, _colors, 1e-4, 1e-4))
            << "Forward pass mismatch for SH degree " << sh_degree
            << "\nMax diff: " << (colors - _colors).abs().max().item<float>();

        // Check that colors are in a reasonable range
        auto min_val = colors.min().item<float>();
        auto max_val = colors.max().item<float>();
        EXPECT_GT(max_val, -10.0f) << "Colors too small";
        EXPECT_LT(min_val, 10.0f) << "Colors too large";

        // For degree 0, verify that changing coefficients changes output
        if (sh_degree == 0) {
            auto coeffs2 = coeffs.clone();
            coeffs2.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}) *= 2.0f;

            auto colors2 = SphericalHarmonicsFunction::apply(
                sh_degree_tensor, dirs, coeffs2, masks)[0];

            // Output should change when coefficients change
            auto change = (colors2 - colors).abs().mean().item<float>();
            EXPECT_GT(change, 0.1f)
                << "Output should change significantly when degree 0 coefficients are doubled";
        }

        // Test gradients using torch::autograd::grad
        auto v_colors = torch::randn_like(colors);

        // Compute loss
        auto loss = (colors * v_colors).sum();
        std::vector<torch::Tensor> inputs = {coeffs, dirs};
        std::vector<torch::Tensor> outputs = {loss};
        std::vector<torch::Tensor> grad_outputs = {};

        auto grads = torch::autograd::grad(
            outputs,
            inputs,
            grad_outputs,
            /*retain_graph=*/true,
            /*create_graph=*/false,
            /*allow_unused=*/true);

        auto v_coeffs = grads[0];
        auto v_dirs = grads[1];

        // Reference gradients
        auto _loss = (_colors * v_colors).sum();
        auto _grads = torch::autograd::grad(
            {_loss},
            inputs,
            grad_outputs,
            /*retain_graph=*/true,
            /*create_graph=*/false,
            /*allow_unused=*/true);

        auto _v_coeffs = _grads[0];
        auto _v_dirs = _grads[1];

        // Check coefficient gradients
        EXPECT_TRUE(v_coeffs.defined())
            << "No gradient for coeffs at degree " << sh_degree;
        EXPECT_FALSE(v_coeffs.isnan().any().item<bool>())
            << "NaN in coeffs gradient at degree " << sh_degree;
        EXPECT_GT(v_coeffs.abs().max().item<float>(), 0)
            << "Zero gradients for coeffs at degree " << sh_degree;

        EXPECT_TRUE(torch::allclose(v_coeffs, _v_coeffs, 1e-4, 1e-4))
            << "Coefficient gradients mismatch for SH degree " << sh_degree
            << "\nMax diff: " << (v_coeffs - _v_coeffs).abs().max().item<float>();

        // For degree > 0, we should have gradients w.r.t. directions
        if (sh_degree > 0) {
            EXPECT_TRUE(v_dirs.defined())
                << "No gradient for dirs at degree " << sh_degree;
            EXPECT_FALSE(v_dirs.isnan().any().item<bool>())
                << "NaN in dirs gradient at degree " << sh_degree;

            // For higher degrees, direction gradients should be non-zero
            auto dirs_grad_norm = v_dirs.abs().max().item<float>();
            EXPECT_GT(dirs_grad_norm, 1e-6f)
                << "Direction gradients too small for degree " << sh_degree;

            EXPECT_TRUE(torch::allclose(v_dirs, _v_dirs, 1e-4, 1e-4))
                << "Direction gradients mismatch for SH degree " << sh_degree
                << "\nMax diff: " << (v_dirs - _v_dirs).abs().max().item<float>();
        } else {
            // For degree 0, direction gradients should be zero or very small
            if (v_dirs.defined() && _v_dirs.defined()) {
                EXPECT_TRUE(torch::allclose(v_dirs, _v_dirs, 1e-4, 1e-4))
                    << "Direction gradients should match for degree 0";
            }
        }

        // Verify gradient magnitude is reasonable
        auto grad_norm = v_coeffs.norm().item<float>();
        auto v_norm = v_colors.norm().item<float>();
        EXPECT_GT(grad_norm, 0) << "Gradient norm is zero";
        EXPECT_LT(grad_norm / v_norm, 1e3) << "Gradient norm is too large relative to v_colors";
    }
}

// Test tile intersection matching Python test
TEST_F(BasicPyTorchDataTest, TileIntersectionTest) {
    torch::manual_seed(42);

    const int C = 3, N = 1000;
    const int width = 40, height = 60;
    const int tile_size = 16;
    const int tile_width = (width + tile_size - 1) / tile_size;
    const int tile_height = (height + tile_size - 1) / tile_size;

    auto means2d = torch::randn({C, N, 2}, device) * width;
    auto radii = torch::randint(0, width, {C, N, 2},
                                torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto depths = torch::rand({C, N}, device);

    // gsplat implementation
    auto [tiles_per_gauss, isect_ids, flatten_ids] = gsplat::intersect_tile(
        means2d, radii, depths, {}, {}, C, tile_size, tile_width, tile_height, true);

    auto isect_offsets = gsplat::intersect_offset(isect_ids, C, tile_width, tile_height);

    // Reference implementation
    auto [_tiles_per_gauss, _isect_ids, _flatten_ids] = reference::isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, true);

    // Convert to same dtype before comparison
    tiles_per_gauss = tiles_per_gauss.to(torch::kInt64);
    _tiles_per_gauss = _tiles_per_gauss.to(torch::kInt64);

    isect_ids = isect_ids.to(torch::kInt64);
    _isect_ids = _isect_ids.to(torch::kInt64);

    flatten_ids = flatten_ids.to(torch::kInt64);
    _flatten_ids = _flatten_ids.to(torch::kInt64);

    // Compare
    assertTensorClose(tiles_per_gauss, _tiles_per_gauss, 0, 0, "tiles_per_gauss");
    assertTensorClose(isect_ids, _isect_ids, 0, 0, "isect_ids");
    assertTensorClose(flatten_ids, _flatten_ids, 0, 0, "flatten_ids");
}

// Test rasterization for 3DGS
TEST_F(BasicPyTorchDataTest, Rasterization3DGSTest) {
    torch::manual_seed(42);

    // For 3DGS rasterization, we need to use the full pipeline
    // similar to how it's done in rasterizer.cpp

    const int N = test_data.means.size(0);
    const int C = 1; // Test with single camera
    const int width = test_data.width;
    const int height = test_data.height;
    const int tile_size = 16;

    // Prepare data
    auto means = test_data.means.clone().set_requires_grad(true);
    auto quats = test_data.quats.clone().set_requires_grad(true);
    auto scales = test_data.scales.clone() * 0.1f; // Scale down for stability
    scales.set_requires_grad(true);
    auto opacities = test_data.opacities.clone().set_requires_grad(true);

    // Single camera
    auto viewmat = test_data.viewmats[0].unsqueeze(0).clone().set_requires_grad(true);
    auto K = test_data.Ks[0].unsqueeze(0);

    // Background
    auto background = torch::rand({1, 3}, device).set_requires_grad(true);

    // Step 1: Projection
    auto proj_settings = torch::tensor({(float)width, (float)height,
                                        0.3f, 0.01f, 10000.0f, 0.0f, 1.0f},
                                       device);

    auto proj_outputs = ProjectionFunction::apply(
        means, quats, scales, opacities, viewmat, K, proj_settings);

    auto radii = proj_outputs[0];
    auto means2d = proj_outputs[1];
    auto depths = proj_outputs[2];
    auto conics = proj_outputs[3];
    auto compensations = proj_outputs[4];

    // Step 2: Get colors (simple test - use random colors)
    auto colors = torch::rand({1, N, 3}, device).set_requires_grad(true);

    // Step 3: Apply opacity with compensations
    auto final_opacities = opacities.unsqueeze(0) * compensations;

    // Step 4: Tile intersection
    const int tile_width = (width + tile_size - 1) / tile_size;
    const int tile_height = (height + tile_size - 1) / tile_size;

    auto [tiles_per_gauss, isect_ids, flatten_ids] = gsplat::intersect_tile(
        means2d, radii, depths, {}, {}, 1, tile_size, tile_width, tile_height, true);

    auto isect_offsets = gsplat::intersect_offset(isect_ids, 1, tile_width, tile_height);
    isect_offsets = isect_offsets.reshape({1, tile_height, tile_width});

    // Step 5: Rasterization using RasterizationFunction
    auto raster_settings = torch::tensor({(float)width, (float)height, (float)tile_size}, device);

    auto raster_outputs = RasterizationFunction::apply(
        means2d, conics, colors, final_opacities, background,
        isect_offsets, flatten_ids, raster_settings);

    auto rendered_image = raster_outputs[0];
    auto rendered_alpha = raster_outputs[1];

    // Check output shapes
    EXPECT_EQ(rendered_image.sizes(), torch::IntArrayRef({1, height, width, 3}));
    EXPECT_EQ(rendered_alpha.sizes(), torch::IntArrayRef({1, height, width, 1}));

    // Check values are valid
    auto img_min = rendered_image.min();
    auto img_max = rendered_image.max();
    EXPECT_GE(img_min.item<float>(), 0.0f) << "Negative colors in rendered image";
    EXPECT_LE(img_max.item<float>(), 1.0f) << "Colors > 1 in rendered image";

    auto alpha_min = rendered_alpha.min();
    auto alpha_max = rendered_alpha.max();
    EXPECT_GE(alpha_min.item<float>(), 0.0f) << "Negative alpha values";
    EXPECT_LE(alpha_max.item<float>(), 1.0f) << "Alpha > 1";

    // Test backward pass
    auto v_image = torch::randn_like(rendered_image);
    auto v_alpha = torch::randn_like(rendered_alpha);

    auto loss = (rendered_image * v_image).sum() + (rendered_alpha * v_alpha).sum();
    loss.backward();

    // Check gradients exist
    EXPECT_TRUE(means.grad().defined());
    EXPECT_TRUE(quats.grad().defined());
    EXPECT_TRUE(scales.grad().defined());
    EXPECT_TRUE(opacities.grad().defined());
    EXPECT_TRUE(colors.grad().defined());
    EXPECT_TRUE(background.grad().defined());

    // Check gradients are valid
    EXPECT_FALSE(means.grad().isnan().any().item<bool>());
    EXPECT_FALSE(quats.grad().isnan().any().item<bool>());
    EXPECT_FALSE(scales.grad().isnan().any().item<bool>());
    EXPECT_FALSE(opacities.grad().isnan().any().item<bool>());
    EXPECT_FALSE(colors.grad().isnan().any().item<bool>());
    EXPECT_FALSE(background.grad().isnan().any().item<bool>());

    // Check some gradients are non-zero
    EXPECT_GT(colors.grad().abs().max().item<float>(), 0) << "Color gradients are all zero";
}