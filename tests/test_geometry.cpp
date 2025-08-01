#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>
#include <numbers> // std::numbers
#include <random>

#include "geometry/euclidean_transform.hpp" // Adjust path as needed

constexpr int RANDOM_SEED = 8128;
using namespace gs::geometry;

class EuclideanTransformTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random number generator with a fixed seed for reproducible tests
        rng.seed(RANDOM_SEED);
        angle_dist = std::uniform_real_distribution<float>(-std::numbers::pi, std::numbers::pi);
        translation_dist = std::uniform_real_distribution<float>(-10.0f, 10.0f);
    }

    // Helper function to check if two matrices are approximately equal
    bool matricesEqual(const glm::mat4& a, const glm::mat4& b, float tolerance = 1e-5f) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (std::abs(a[i][j] - b[i][j]) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    // Helper function to check if two vectors are approximately equal
    bool vectorsEqual(const glm::vec3& a, const glm::vec3& b, float tolerance = 1e-5f) {
        return glm::length(a - b) < tolerance;
    }

    // Helper function to check if two quaternions are approximately equal
    bool quaternionsEqual(const glm::quat& a, const glm::quat& b, float tolerance = 1e-5f) {
        // Quaternions q and -q represent the same rotation
        return (glm::length(glm::vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w)) < tolerance) ||
               (glm::length(glm::vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w)) < tolerance);
    }

    // Generate random Euler angles
    glm::vec3 generateRandomAngles() {
        return glm::vec3(angle_dist(rng), angle_dist(rng), angle_dist(rng));
    }

    // Generate random translation vector
    glm::vec3 generateRandomTranslation() {
        return glm::vec3(translation_dist(rng), translation_dist(rng), translation_dist(rng));
    }

    std::mt19937 rng;
    std::uniform_real_distribution<float> angle_dist;
    std::uniform_real_distribution<float> translation_dist;
};

// Test 1: Matrix roundtrip conversion (mat4 -> EuclideanTransform -> mat4)
TEST_F(EuclideanTransformTest, MatrixRoundtripConversion) {
    const int num_tests = 10;

    for (int i = 0; i < num_tests; ++i) {
        // Generate random angles and translation
        glm::vec3 angles = generateRandomAngles();
        glm::vec3 translation = generateRandomTranslation();

        // Create original transformation matrix
        glm::mat4 rotation_mat = glm::rotate(glm::mat4(1.0f), angles.z, glm::vec3(0, 0, 1));
        rotation_mat = glm::rotate(rotation_mat, angles.y, glm::vec3(0, 1, 0));
        rotation_mat = glm::rotate(rotation_mat, angles.x, glm::vec3(1, 0, 0));
        glm::mat4 original_matrix = glm::translate(rotation_mat, translation);

        // Convert to EuclideanTransform and back to matrix
        EuclideanTransform transform(original_matrix);
        glm::mat4 reconstructed_matrix = transform.toMat4();

        // Check if matrices are equal
        EXPECT_TRUE(matricesEqual(original_matrix, reconstructed_matrix))
            << "Test " << i << " failed: Matrix roundtrip conversion mismatch\n"
            << "Original matrix differs from reconstructed matrix";
    }
}

// Test 2: Inverse transformation correctness
TEST_F(EuclideanTransformTest, InverseTransformation) {
    const int num_tests = 10;

    for (int i = 0; i < num_tests; ++i) {
        // Generate random angles and translation
        glm::vec3 angles = generateRandomAngles();
        glm::vec3 translation = generateRandomTranslation();

        // Create EuclideanTransform
        EuclideanTransform transform(angles.x, angles.y, angles.z,
                                     translation.x, translation.y, translation.z);

        // Get the inverse
        EuclideanTransform inverse = transform.inv();

        // Expected inverse calculation:
        // For a transformation T = [R | t], inverse is [R^T | -R^T * t]
        glm::quat rotation = transform.getRotation();
        glm::vec3 trans = transform.getTranslation();

        // Expected inverse rotation (conjugate for unit quaternions)
        glm::quat expected_inv_rotation = glm::conjugate(rotation);

        // Expected inverse translation: -R^T * t
        glm::vec3 expected_inv_translation = -(expected_inv_rotation * trans);

        // Check rotation
        EXPECT_TRUE(quaternionsEqual(inverse.getRotation(), expected_inv_rotation))
            << "Test " << i << " failed: Inverse rotation mismatch";

        // Check translation
        EXPECT_TRUE(vectorsEqual(inverse.getTranslation(), expected_inv_translation))
            << "Test " << i << " failed: Inverse translation mismatch";

        // Additional verification: T * T^(-1) should be identity
        EuclideanTransform identity_check = transform * inverse;
        glm::mat4 identity_matrix = identity_check.toMat4();
        glm::mat4 expected_identity = glm::mat4(1.0f);

        EXPECT_TRUE(matricesEqual(identity_matrix, expected_identity))
            << "Test " << i << " failed: T * T^(-1) is not identity";

        // Also check T^(-1) * T should be identity
        EuclideanTransform identity_check2 = inverse * transform;
        glm::mat4 identity_matrix2 = identity_check2.toMat4();

        EXPECT_TRUE(matricesEqual(identity_matrix2, expected_identity))
            << "Test " << i << " failed: T^(-1) * T is not identity";
    }
}

// Test 3: Constructor consistency tests
TEST_F(EuclideanTransformTest, ConstructorConsistency) {
    const int num_tests = 10;

    for (int i = 0; i < num_tests; ++i) {
        glm::vec3 angles = generateRandomAngles();
        glm::vec3 translation = generateRandomTranslation();

        // Create transform using Euler angles constructor
        EuclideanTransform transform1(angles.x, angles.y, angles.z,
                                      translation.x, translation.y, translation.z);

        // Create the same transform using quaternion + translation constructor
        glm::quat rotation = glm::quat(angles);
        EuclideanTransform transform2(rotation, translation);

        // Both should produce the same matrix
        glm::mat4 matrix1 = transform1.toMat4();
        glm::mat4 matrix2 = transform2.toMat4();

        EXPECT_TRUE(matricesEqual(matrix1, matrix2))
            << "Test " << i << " failed: Constructor consistency mismatch";
    }
}

// Test 4: Point and vector transformation
TEST_F(EuclideanTransformTest, PointAndVectorTransformation) {
    const int num_tests = 10;

    for (int i = 0; i < num_tests; ++i) {
        glm::vec3 angles = generateRandomAngles();
        glm::vec3 translation = generateRandomTranslation();

        EuclideanTransform transform(angles.x, angles.y, angles.z,
                                     translation.x, translation.y, translation.z);

        // Test point
        glm::vec3 test_point = generateRandomTranslation(); // Reuse for random point
        glm::vec3 transformed_point = transform.transformPoint(test_point);

        // Verify using matrix multiplication
        glm::mat4 matrix = transform.toMat4();
        glm::vec4 homogeneous_point(test_point, 1.0f);
        glm::vec4 expected_transformed = matrix * homogeneous_point;
        glm::vec3 expected_point(expected_transformed.x, expected_transformed.y, expected_transformed.z);

        EXPECT_TRUE(vectorsEqual(transformed_point, expected_point))
            << "Test " << i << " failed: Point transformation mismatch";

        // Test vector (should only apply rotation)
        glm::vec3 test_vector = generateRandomTranslation(); // Reuse for random vector
        glm::vec3 transformed_vector = transform.transformVector(test_vector);

        // Expected: only rotation applied
        glm::vec3 expected_vector = transform.getRotation() * test_vector;

        EXPECT_TRUE(vectorsEqual(transformed_vector, expected_vector))
            << "Test " << i << " failed: Vector transformation mismatch";
    }
}

// Test 5: Composition operator
TEST_F(EuclideanTransformTest, CompositionOperator) {
    const int num_tests = 10;

    for (int i = 0; i < num_tests; ++i) {
        // Create two random transforms
        glm::vec3 angles1 = generateRandomAngles();
        glm::vec3 translation1 = generateRandomTranslation();
        glm::vec3 angles2 = generateRandomAngles();
        glm::vec3 translation2 = generateRandomTranslation();

        EuclideanTransform transform1(angles1.x, angles1.y, angles1.z,
                                      translation1.x, translation1.y, translation1.z);
        EuclideanTransform transform2(angles2.x, angles2.y, angles2.z,
                                      translation2.x, translation2.y, translation2.z);

        // Compose using operator*
        EuclideanTransform composed = transform1 * transform2;

        // Compose using matrix multiplication
        glm::mat4 matrix1 = transform1.toMat4();
        glm::mat4 matrix2 = transform2.toMat4();
        glm::mat4 expected_composed_matrix = matrix1 * matrix2;

        glm::mat4 actual_composed_matrix = composed.toMat4();

        EXPECT_TRUE(matricesEqual(actual_composed_matrix, expected_composed_matrix))
            << "Test " << i << " failed: Composition operator mismatch";
    }
}

// Test 6: Identity transformation
TEST_F(EuclideanTransformTest, IdentityTransformation) {
    EuclideanTransform identity;
    glm::mat4 identity_matrix = identity.toMat4();
    glm::mat4 expected_identity = glm::mat4(1.0f);

    EXPECT_TRUE(matricesEqual(identity_matrix, expected_identity))
        << "Default constructor should create identity transformation";

    // Test that identity transformation doesn't change points
    glm::vec3 test_point(1.0f, 2.0f, 3.0f);
    glm::vec3 transformed = identity.transformPoint(test_point);

    EXPECT_TRUE(vectorsEqual(transformed, test_point))
        << "Identity transformation should not change points";
}

// Test 7: Euler angles roundtrip conversion
TEST_F(EuclideanTransformTest, EulerAnglesRoundtrip) {
    const int num_tests = 10;

    for (int i = 0; i < num_tests; ++i) {
        // Generate random Euler angles
        glm::vec3 original_angles = generateRandomAngles();

        // Create EuclideanTransform with zero translation
        EuclideanTransform transform(original_angles.x, original_angles.y, original_angles.z,
                                     0.0f, 0.0f, 0.0f);

        // Get the Euler angles back
        glm::vec3 retrieved_angles = transform.getEulerAngles();

        // The tricky part: Euler angles can have multiple representations
        // We need to check if the rotations are equivalent, not just the angles

        // Method 1: Check if the rotation matrices are the same
        glm::mat4 original_rotation = glm::rotate(glm::mat4(1.0f), original_angles.z, glm::vec3(0, 0, 1));
        original_rotation = glm::rotate(original_rotation, original_angles.y, glm::vec3(0, 1, 0));
        original_rotation = glm::rotate(original_rotation, original_angles.x, glm::vec3(1, 0, 0));

        glm::mat4 retrieved_rotation = glm::rotate(glm::mat4(1.0f), retrieved_angles.z, glm::vec3(0, 0, 1));
        retrieved_rotation = glm::rotate(retrieved_rotation, retrieved_angles.y, glm::vec3(0, 1, 0));
        retrieved_rotation = glm::rotate(retrieved_rotation, retrieved_angles.x, glm::vec3(1, 0, 0));

        EXPECT_TRUE(matricesEqual(original_rotation, retrieved_rotation))
            << "Test " << i << " failed: Euler angles roundtrip - rotation matrices don't match\n"
            << "Original angles: (" << original_angles.x << ", " << original_angles.y << ", " << original_angles.z << ")\n"
            << "Retrieved angles: (" << retrieved_angles.x << ", " << retrieved_angles.y << ", " << retrieved_angles.z << ")";
    }
}

// Main function for running tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}