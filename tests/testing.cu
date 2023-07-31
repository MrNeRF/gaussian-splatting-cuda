#include "general_utils.cuh"
#include <gtest/gtest.h>

TEST(inverse_sigmoid_test, returns_expected_values) {
    // Test input value of 0.5
    {
        auto input = torch::tensor(0.5);
        auto output = inverse_sigmoid(input);
        EXPECT_DOUBLE_EQ(output.item<double>(), 0.0);
    }

    // Test input value of 0.9
    {
        auto input = torch::tensor(0.9);
        auto output = inverse_sigmoid(input);
        EXPECT_NEAR(output.item<double>(), 2.197224577, 1e-6);
    }

    // Test input value of 0.1
    {
        auto input = torch::tensor(0.1);
        auto output = inverse_sigmoid(input);
        EXPECT_NEAR(output.item<double>(), -2.197224577, 1e-6);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}