// Copyright (c) 2023 Janusch Patas.

#include "debug_utils.cuh"
#include "render_utils.cuh"
#include "serialization.h"
#include <gtest/gtest.h>
#include <tuple>

class RasterizeGaussiansTest : public ::testing::Test {
protected:
    torch::Tensor dL_means2D;
    torch::Tensor dL_colors_precomp;
    torch::Tensor dL_opacities;
    torch::Tensor dL_means3D;
    torch::Tensor dL_cov3Ds_precomp;
    torch::Tensor dL_sh;
    torch::Tensor dL_scales;
    torch::Tensor dL_rotations;
    torch::Tensor background;
    torch::Tensor means3D;
    torch::Tensor radii;
    torch::Tensor colors;
    torch::Tensor scales;
    torch::Tensor rotations;
    float scale_modifier;
    torch::Tensor cov3D_precomp;
    torch::Tensor viewmatrix;
    torch::Tensor projmatrix;
    float tan_fovx;
    float tan_fovy;
    torch::Tensor dL_dout_color;
    torch::Tensor sh;
    int degree;
    torch::Tensor campos;
    torch::Tensor geomBuffer;
    int R;
    torch::Tensor binningBuffer;
    torch::Tensor imageBuffer;

    // Load data before each test
    virtual void SetUp() {
        loadFunctionData("rasterize_backward_test_data.dat", dL_means2D, dL_colors_precomp, dL_opacities, dL_means3D,
                         dL_cov3Ds_precomp, dL_sh, dL_scales, dL_rotations, background,
                         means3D, radii, colors, scales, rotations,
                         scale_modifier, cov3D_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy, dL_dout_color, sh,
                         degree, campos, geomBuffer, R, binningBuffer, imageBuffer);
    }
};

TEST_F(RasterizeGaussiansTest, CompareOutputs) {
    auto [grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations] = RasterizeGaussiansBackwardCUDA(
        background, means3D, radii, colors, scales, rotations, scale_modifier,
        cov3D_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy,
        dL_dout_color, sh, degree, campos, geomBuffer, R, binningBuffer,
        imageBuffer, false);

    // return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
    const double epsilon = 1e-3;
    EXPECT_TRUE(torch::allclose(grad_means2D, dL_means2D, epsilon));
    EXPECT_TRUE(torch::allclose(grad_colors_precomp, dL_colors_precomp, epsilon));
    EXPECT_TRUE(torch::allclose(grad_opacities, dL_opacities, epsilon));
    EXPECT_TRUE(torch::allclose(grad_means3D, dL_means3D, epsilon));
    EXPECT_TRUE(torch::allclose(grad_cov3Ds_precomp, dL_cov3Ds_precomp, epsilon));
    EXPECT_TRUE(torch::allclose(grad_sh, dL_sh, epsilon));
    EXPECT_TRUE(torch::allclose(grad_scales, dL_scales, epsilon));
    EXPECT_TRUE(torch::allclose(grad_rotations, dL_rotations, epsilon));
}
