// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#include "camera_utils.cuh"
#include <gtest/gtest.h>

// class TransformMatrixTest : public ::testing::Test {
// protected:
//     Eigen::Matrix3d R;
//     Eigen::Vector3d t;

//     void SetUp() override {
//         // Initialize R and t with some values.
//         R << 0.707, -0.707, 0,
//              0.707, 0.707, 0,
//              0, 0, 1;
//         t << 1, 2, 3;
//     }
// };

// TEST_F(TransformMatrixTest, getWorld2ViewTest) {
//     Eigen::Matrix4d W2V = getWorld2View(R, t);

//     EXPECT_EQ(W2V.rows(), 4);
//     EXPECT_EQ(W2V.cols(), 4);
//     EXPECT_DOUBLE_EQ(W2V(0, 0), 0.707);
//     EXPECT_DOUBLE_EQ(W2V(0, 1), 0.707);
//     EXPECT_DOUBLE_EQ(W2V(0, 2), 0);
//     EXPECT_DOUBLE_EQ(W2V(0, 3), -1.414);
//     EXPECT_DOUBLE_EQ(W2V(1, 0), -0.707);
//     EXPECT_DOUBLE_EQ(W2V(1, 1), 0.707);
//     EXPECT_DOUBLE_EQ(W2V(1, 2), 0);
//     EXPECT_DOUBLE_EQ(W2V(1, 3), -2.828);
//     EXPECT_DOUBLE_EQ(W2V(2, 0), 0);
//     EXPECT_DOUBLE_EQ(W2V(2, 1), 0);
//     EXPECT_DOUBLE_EQ(W2V(2, 2), 1);
//     EXPECT_DOUBLE_EQ(W2V(2, 3), -3);
//     EXPECT_DOUBLE_EQ(W2V(3, 0), 0);
//     EXPECT_DOUBLE_EQ(W2V(3, 1), 0);
//     EXPECT_DOUBLE_EQ(W2V(3, 2), 0);
//     EXPECT_DOUBLE_EQ(W2V(3, 3), 1);
// }

// TEST_F(TransformMatrixTest, getWorld2View2Test) {
//     Eigen::Matrix4d W2V2 = getWorld2View2(R, t);

//     // Now validate the output. This could be by checking the size, properties or
//     // specific elements of the matrix. Here is an example:
//     EXPECT_EQ(W2V2.rows(), 4);
//     EXPECT_EQ(W2V2.cols(), 4);
//     EXPECT_DOUBLE_EQ(W2V2(0, 0), 1.0);
//     EXPECT_DOUBLE_EQ(W2V2(0, 1), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(0, 2), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(0, 3), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(1, 0), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(1, 1), 1.0);
//     EXPECT_DOUBLE_EQ(W2V2(1, 2), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(1, 3), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(2, 0), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(2, 1), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(2, 2), 1.0);
//     EXPECT_DOUBLE_EQ(W2V2(2, 3), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(3, 0), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(3, 1), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(3, 2), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(3, 3), 1.0);

//     Eigen::Vector3d translate(2, 3, 4);
//     float scale = 2.0;
//     W2V2 = getWorld2View2(R, t, translate, scale);

//     // Now validate the output again. Here is an example:
//     EXPECT_EQ(W2V2.rows(), 4);
//     EXPECT_EQ(W2V2.cols(), 4);
//     EXPECT_DOUBLE_EQ(W2V2(0, 0), 1.414);
//     EXPECT_DOUBLE_EQ(W2V2(0, 1), -1.414);
//     EXPECT_DOUBLE_EQ(W2V2(0, 2), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(0, 3), -1.414);
//     EXPECT_DOUBLE_EQ(W2V2(1, 0), 1.414);
//     EXPECT_DOUBLE_EQ(W2V2(1, 1), 1.414);
//     EXPECT_DOUBLE_EQ(W2V2(1, 2), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(1, 3), -2.828);
//     EXPECT_DOUBLE_EQ(W2V2(2, 0), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(2, 1), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(2, 2), 2.0);
//     EXPECT_DOUBLE_EQ(W2V2(2, 3), -3.0);
//     EXPECT_DOUBLE_EQ(W2V2(3, 0), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(3, 1), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(3, 2), 0.0);
//     EXPECT_DOUBLE_EQ(W2V2(3, 3), 1.0);
// }

class ProjectionMatrixTest : public ::testing::Test {
protected:
    double znear, zfar, fovX, fovY;
    double pixels;

    void SetUp() override {
        znear = 0.1;
        zfar = 100.0;
        fovX = M_PI / 2.0; // 90 degrees in radians
        fovY = M_PI / 3.0; // 60 degrees in radians
        pixels = 1080.0;
    }
};

TEST_F(ProjectionMatrixTest, getProjectionMatrixTest) {
    Eigen::Matrix4d P = getProjectionMatrix(znear, zfar, fovX, fovY);

    // Validate the output
    EXPECT_EQ(P.rows(), 4);
    EXPECT_EQ(P.cols(), 4);

    // Calculate expected values based on the provided implementation of getProjectionMatrix
    double tanHalfFovY = std::tan((fovY / 2));
    double tanHalfFovX = std::tan((fovX / 2));
    double top = tanHalfFovY * znear;
    double bottom = -top;
    double right = tanHalfFovX * znear;
    double left = -right;
    double z_sign = 1.0;

    EXPECT_DOUBLE_EQ(P(0, 0), 2.0 * znear / (right - left));
    EXPECT_DOUBLE_EQ(P(1, 1), 2.0 * znear / (top - bottom));
    EXPECT_DOUBLE_EQ(P(0, 2), (right + left) / (right - left));
    EXPECT_DOUBLE_EQ(P(1, 2), (top + bottom) / (top - bottom));
    EXPECT_DOUBLE_EQ(P(3, 2), z_sign);
    EXPECT_DOUBLE_EQ(P(2, 2), z_sign * zfar / (zfar - znear));
    EXPECT_DOUBLE_EQ(P(2, 3), -(zfar * znear) / (zfar - znear));
}

TEST_F(ProjectionMatrixTest, fov2focalTest) {
    double focal = fov2focal(fovX, pixels);

    // Validate the output
    // Given the above implementation of fov2focal, the expected value is:
    double expected = pixels / (2 * std::tan(fovX / 2.0));
    EXPECT_DOUBLE_EQ(focal, expected);
}

TEST_F(ProjectionMatrixTest, focal2fovTest) {
    double focal = fov2focal(fovX, pixels);
    double recoveredFov = focal2fov(focal, pixels);

    // Validate the output
    // The fov value should be recovered
    EXPECT_DOUBLE_EQ(recoveredFov, fovX);
}

class RotationTest : public ::testing::Test {
protected:
    Eigen::Quaterniond q;
    Eigen::Matrix3d R;

    void SetUp() override {
        // Initialize a Quaternion
        q = Eigen::Quaterniond::UnitRandom();

        // And its corresponding rotation matrix
        R = q.toRotationMatrix();
    }
};

TEST_F(RotationTest, qvec2rotmatTest) {
    Eigen::Matrix3d R_test = qvec2rotmat(q);
    // Validate the output
    // The output should be a 3x3 rotation matrix which is equivalent to the input quaternion
    EXPECT_EQ(R_test.rows(), 3);
    EXPECT_EQ(R_test.cols(), 3);
    EXPECT_TRUE(R.isApprox(R_test, 1e-10)); // Compare with a small error tolerance
}

TEST_F(RotationTest, rotmat2qvecTest) {
    Eigen::Quaterniond q_test = rotmat2qvec(R);
    // Validate the output
    // The output should be a quaternion which is equivalent to the input rotation matrix
    // Note: two quaternions q and -q represent the same rotation, hence the need for the .isApprox check
    EXPECT_TRUE(q.coeffs().isApprox(q_test.coeffs(), 1e-10) ||
                q.coeffs().isApprox((q_test.coeffs() * -1.0), 1e-10));
}

TEST_F(RotationTest, roundTripTest) {
    // Perform a round trip conversion: quaternion -> rotation matrix -> quaternion
    Eigen::Quaterniond q_roundtrip = rotmat2qvec(qvec2rotmat(q));
    // The round trip should preserve the quaternion
    EXPECT_TRUE(q.coeffs().isApprox(q_roundtrip.coeffs(), 1e-10) ||
                q.coeffs().isApprox((q_roundtrip.coeffs() * -1.0), 1e-10));

    // And in the other direction: rotation matrix -> quaternion -> rotation matrix
    Eigen::Matrix3d R_roundtrip = qvec2rotmat(rotmat2qvec(R));
    // The round trip should preserve the rotation matrix
    EXPECT_TRUE(R.isApprox(R_roundtrip, 1e-10));
}