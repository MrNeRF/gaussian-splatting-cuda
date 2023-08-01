// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#include "general_utils.cuh"
#include <gtest/gtest.h>

/**
 * The StripLowerDiagTest class is a Google Test fixture that tests the strip_lowerdiag function.
 * The CheckStripLowerDiag member function tests whether the function correctly extracts the lower diagonal elements of a tensor.
 * The INSTANTIATE_TEST_SUITE_P macro instantiates the StripLowerDiagTest class with three different test cases.
 */
class StripLowerDiagTest : public ::testing::TestWithParam<torch::Tensor> {
};

TEST_P(StripLowerDiagTest, CheckStripLowerDiag) {
    auto input_tensor = GetParam();

    auto result_tensor = strip_lowerdiag(input_tensor);

    // Confirm that the result tensor has the right size
    ASSERT_EQ(result_tensor.size(0), input_tensor.size(0));
    ASSERT_EQ(result_tensor.size(1), 6);

    // Confirm that the lower diagonal elements have been correctly extracted
    for (int i = 0; i < input_tensor.size(0); ++i) {
        EXPECT_FLOAT_EQ(result_tensor[i][0].item<float>(), input_tensor[i][0][0].item<float>());
        EXPECT_FLOAT_EQ(result_tensor[i][1].item<float>(), input_tensor[i][0][1].item<float>());
        EXPECT_FLOAT_EQ(result_tensor[i][2].item<float>(), input_tensor[i][0][2].item<float>());
        EXPECT_FLOAT_EQ(result_tensor[i][3].item<float>(), input_tensor[i][1][1].item<float>());
        EXPECT_FLOAT_EQ(result_tensor[i][4].item<float>(), input_tensor[i][1][2].item<float>());
        EXPECT_FLOAT_EQ(result_tensor[i][5].item<float>(), input_tensor[i][2][2].item<float>());
    }
}

INSTANTIATE_TEST_SUITE_P(
    StripLowerDiagTest,
    StripLowerDiagTest,
    ::testing::Values(
        torch::randn({1, 3, 3}), // Test case 1: single 3x3 matrix
        torch::randn({2, 3, 3}), // Test case 2: two 3x3 matrices
        torch::randn({10, 3, 3}) // Test case 3: ten 3x3 matrices
        ));

/**
 * The LearningRateTest struct is a test fixture that is used to test the Expon_lr_func class. The LearningRateTest test case
 * checks the learning rate at the start, middle, and end of training for different sets of parameters.
 *
 * The INSTANTIATE_TEST_SUITE_P macro instantiates the LearningRateTest test case with different sets of parameters.
 */
struct LearningRateTest : public ::testing::TestWithParam<std::tuple<double, double, int64_t, double, int64_t>> {
};

TEST_P(LearningRateTest, CheckLearningRate) {
    double lr_init;
    double lr_final;
    int64_t lr_delay_steps;
    double lr_delay_mult;
    int64_t max_steps;

    std::tie(lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps) = GetParam();

    Expon_lr_func lr_func(lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps);

    // Check learning rate at the start of training
    EXPECT_NEAR(lr_func(0), lr_init, 0.0001);

    // Check learning rate in the middle of training
    double mid_step = max_steps / 2.0;
    double mid_lr = std::sqrt(lr_init * lr_final);
    EXPECT_NEAR(lr_func(mid_step), mid_lr, 0.0001);

    // Check learning rate at the end of training
    EXPECT_NEAR(lr_func(max_steps), lr_final, 0.0001);
}

/**
 * Instantiates a test suite for the ExponLrFuncTest and LearningRateTest classes with multiple test cases.
 *
 * @param ExponLrFuncTest The test suite for the ExponLrFunc class.
 * @param LearningRateTest The test suite for the LearningRate class.
 * @param Values A tuple of test cases, each containing five values: learning rate, weight decay, step size, gamma, and max iterations.
 */
INSTANTIATE_TEST_SUITE_P(
    ExponLrFuncTest,
    LearningRateTest,
    ::testing::Values(
        std::make_tuple(0.1, 0.01, 0, 1.0, 1000),   // Test case 1
        std::make_tuple(0.1, 0.01, 200, 0.5, 1000), // Test case 2
        std::make_tuple(0.1, 0.01, 0, 1.0, 2000)    // Test case 3
        ));

struct InverseSigmoidTest : public ::testing::TestWithParam<std::tuple<torch::Tensor, torch::Tensor>> {
};

TEST_P(InverseSigmoidTest, CheckInverseSigmoid) {
    torch::Tensor input;
    torch::Tensor expected_output;

    std::tie(input, expected_output) = GetParam();

    torch::Tensor output = inverse_sigmoid(input);

    ASSERT_TRUE(torch::allclose(output, expected_output, 1e-5));
}

INSTANTIATE_TEST_SUITE_P(
    InverseSigmoidFuncTest,
    InverseSigmoidTest,
    ::testing::Values(
        std::make_tuple(torch::tensor(0.5), torch::log(torch::tensor(0.5) / torch::tensor(0.5))),
        std::make_tuple(torch::tensor(0.1), torch::log(torch::tensor(0.1) / torch::tensor(0.9))),
        std::make_tuple(torch::tensor(0.9), torch::log(torch::tensor(0.9) / torch::tensor(0.1)))));

/**
 * The BuildRotationTest class is defined to test the build_rotation function. The CheckBuildRotation test case
 * checks if the output tensor has the correct size and if the output tensor is correct for the identity quaternion.
 * The INSTANTIATE_TEST_SUITE_P macro instantiates the BuildRotationTest class with two test cases: the identity quaternion
 * and a random quaternion.
 */
class BuildRotationTest : public ::testing::TestWithParam<torch::Tensor> {
};

TEST_P(BuildRotationTest, CheckBuildRotation) {
    auto input_tensor = GetParam();

    auto result_tensor = build_rotation(input_tensor);

    // Confirm that the result tensor has the right size
    ASSERT_EQ(result_tensor.size(0), input_tensor.size(0));
    ASSERT_EQ(result_tensor.size(1), 3);
    ASSERT_EQ(result_tensor.size(2), 3);

    // For an identity quaternion [1, 0, 0, 0], the resulting rotation matrix should be the identity matrix
    if (input_tensor.equal(torch::tensor({1.0, 0.0, 0.0, 0.0}).cuda())) {
        EXPECT_TRUE(result_tensor.equal(torch::eye(3).unsqueeze(0).cuda()));
    }
}

INSTANTIATE_TEST_SUITE_P(
    BuildRotationTest,
    BuildRotationTest,
    ::testing::Values(
        torch::tensor({1.0, 0.0, 0.0, 0.0}).unsqueeze(0).cuda(), // identity quaternion
        torch::randn({1, 4}).cuda()                              // random quaternion
        ));

/**
 * The test checks if the output tensor has the correct size.
 *
 * The INSTANTIATE_TEST_SUITE_P macro instantiates the BuildScalingRotationTest class with two different
 * sets of input parameters: identity scaling and rotation, and random scaling and rotation.
 *
 * @param s A tensor representing scaling.
 * @param r A tensor representing rotation.
 * @return void
 */
class BuildScalingRotationTest : public ::testing::TestWithParam<std::tuple<torch::Tensor, torch::Tensor>> {
};

TEST_P(BuildScalingRotationTest, CheckBuildScalingRotation) {
    torch::Tensor s, r;
    std::tie(s, r) = GetParam();

    auto result_tensor = build_scaling_rotation(s, r);

    // Confirm that the result tensor has the right size
    ASSERT_EQ(result_tensor.size(0), s.size(0));
    ASSERT_EQ(result_tensor.size(1), 3);
    ASSERT_EQ(result_tensor.size(2), 3);
}

INSTANTIATE_TEST_SUITE_P(
    BuildScalingRotationTest,
    BuildScalingRotationTest,
    ::testing::Values(
        std::make_tuple(torch::ones({1, 3}).cuda(), torch::tensor({1.0, 0.0, 0.0, 0.0}).unsqueeze(0).cuda()), // identity scaling and rotation
        std::make_tuple(torch::randn({1, 3}).cuda(), torch::randn({1, 4}).cuda())                             // random scaling and rotation
        ));

// TODO: This is failing currently. Need to fix it.
// struct ImageToTorchTest : public ::testing::TestWithParam<std::tuple<int, std::array<int64_t, 3>, int, bool>> {
// };

// TEST_P(ImageToTorchTest, CheckImageToTorch) {
//     int input_value;
//     std::array<int64_t, 3> input_size_array;
//     int resolution_value;
//     bool is_2d;

//     std::tie(input_value, input_size_array, resolution_value, is_2d) = GetParam();

//     std::vector<int64_t> input_size(input_size_array.begin(), input_size_array.end());
//     torch::Tensor input = torch::full(input_size, input_value, torch::kUInt8);
//     std::vector<int64_t> resolution = std::vector<int64_t>(input_size.size(), resolution_value);
//     torch::Tensor expected_output = (input.to(torch::kFloat) / 255.0).permute({1, 2, 0});
//     if(is_2d) {
//         expected_output = expected_output.unsqueeze(-1);
//     }

//     torch::Tensor output = ImageToTorch(input, resolution);

//     ASSERT_TRUE(torch::allclose(output, expected_output, 1e-5));
// }

// INSTANTIATE_TEST_SUITE_P(
//     ImageToTorchFuncTest,
//     ImageToTorchTest,
//     ::testing::Values(
//         // 3D tensor test case
//         std::make_tuple(1, std::array<int64_t, 3>{3, 64, 64}, 64, false),
//         // 2D tensor test case
//         std::make_tuple(1, std::array<int64_t, 3>{64, 64, 1}, 64, true)
//     )
// );
