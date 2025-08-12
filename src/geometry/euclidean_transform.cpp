#include "geometry/euclidean_transform.hpp"

namespace gs {
    namespace geometry {
        EuclideanTransform::EuclideanTransform()
            : m_rotation(glm::identity<glm::quat>()),
              m_translation(0.0f, 0.0f, 0.0f) {
            // Identity quaternion and zero translation
        }

        EuclideanTransform::EuclideanTransform(float x_rad, float y_rad, float z_rad,
                                               float x, float y, float z)
            : m_translation(x, y, z) {
            // Create quaternion from Euler angles using GLM's built-in function
            glm::vec3 eulerAngles(x_rad, y_rad, z_rad);
            m_rotation = glm::quat(eulerAngles);
        }

        EuclideanTransform::EuclideanTransform(const glm::vec3& trans)
            : m_rotation(1.0f, 0.0f, 0.0f, 0.0f),
              m_translation(trans) {
        }

        EuclideanTransform::EuclideanTransform(const glm::quat& rot, const glm::vec3& trans)
            : m_rotation(rot),
              m_translation(trans) {
        }

        EuclideanTransform::EuclideanTransform(const glm::mat4& matrix) {
            // Orthonormalize the input matrix first
            glm::mat4 orthonormalMatrix = OrthonormalizeRotation(matrix);

            // Extract translation from the last column
            m_translation = glm::vec3(orthonormalMatrix[3]);

            // Extract rotation matrix (upper-left 3x3)
            glm::mat3 rotMatrix = glm::mat3(orthonormalMatrix);

            // Convert rotation matrix to quaternion
            m_rotation = glm::quat_cast(rotMatrix);
        }

        glm::mat4 EuclideanTransform::toMat4() const {
            // Create rotation matrix from quaternion
            glm::mat4 rotMatrix = glm::mat4_cast(m_rotation);

            // Create translation matrix
            glm::mat4 transMatrix = glm::translate(glm::mat4(1.0f), m_translation);

            // Combine: T * R (translation then rotation order in matrix multiplication)
            return transMatrix * rotMatrix;
        }

        EuclideanTransform EuclideanTransform::operator*(const EuclideanTransform& other) const {
            // Combine rotations: q1 * q2
            glm::quat newRotation = m_rotation * other.m_rotation;

            // Combine translations: t1 + R1 * t2
            glm::vec3 newTranslation = m_translation + (m_rotation * other.m_translation);

            return EuclideanTransform(newRotation, newTranslation);
        }

        EuclideanTransform& EuclideanTransform::operator*=(const EuclideanTransform& other) {
            *this = *this * other;
            return *this;
        }

        EuclideanTransform EuclideanTransform::inv() const {
            // Inverse rotation is conjugate of quaternion
            glm::quat invRotation = glm::conjugate(m_rotation);

            // Inverse translation: -R^-1 * t = -R* * t
            glm::vec3 invTranslation = -(invRotation * m_translation);

            return EuclideanTransform(invRotation, invTranslation);
        }

        glm::vec3 EuclideanTransform::transformPoint(const glm::vec3& point) const {
            // Apply rotation then translation: R * p + t
            return (m_rotation * point) + m_translation;
        }

        glm::vec3 EuclideanTransform::transformVector(const glm::vec3& vector) const {
            // Apply only rotation: R * v
            return m_rotation * vector;
        }

        glm::mat4 EuclideanTransform::OrthonormalizeRotation(const glm::mat4& matrix) {
            glm::vec3 x = glm::vec3(matrix[0]);
            glm::vec3 y = glm::vec3(matrix[1]);
            glm::vec3 z = glm::vec3(matrix[2]);
            x = glm::normalize(x);
            y = glm::normalize(y - x * glm::dot(x, y));
            z = glm::normalize(glm::cross(x, y));
            glm::mat4 result = glm::mat4(1.0f);
            result[0] = glm::vec4(x, 0.0f);
            result[1] = glm::vec4(y, 0.0f);
            result[2] = glm::vec4(z, 0.0f);
            result[3] = matrix[3];
            return result;
        }

        glm::mat3 EuclideanTransform::getRotationMat() const {
            // Convert quaternion to 3x3 rotation matrix
            return glm::mat3_cast(m_rotation);
        }

        bool EuclideanTransform::isIdentity(float eps) const {
            // Check if translation is near zero
            if (glm::length(m_translation) > eps) {
                return false;
            }

            // Check if rotation is near identity quaternion
            // Identity quaternion is (1, 0, 0, 0) or (-1, 0, 0, 0) due to double cover
            // We can check the angle of rotation: if small, it's near identity

            // Method 1: Check quaternion components directly
            // For identity: w = ±1, x = y = z = 0
            float w_abs = glm::abs(m_rotation.w);
            if (glm::abs(w_abs - 1.0f) > eps) {
                return false;
            }

            glm::vec3 xyz = glm::vec3(m_rotation.x, m_rotation.y, m_rotation.z);
            if (glm::length(xyz) > eps) {
                return false;
            }

            return true;
        }

    } // namespace geometry
} // namespace gs
