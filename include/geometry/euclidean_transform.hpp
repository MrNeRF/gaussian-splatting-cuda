/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace gs {
    namespace geometry {

        class EuclideanTransform {
        private:
            glm::quat m_rotation{};  // Quaternion for rotation
            glm::vec3 m_translation; // Vector for translation

        public:
            // Default constructor - identity transformation
            EuclideanTransform();

            // Constructor from Euler angles (in radians) and translation
            EuclideanTransform(float x_rad, float y_rad, float z_rad,
                               float x, float y, float z);

            // Constructor from translation
            explicit EuclideanTransform(const glm::vec3& trans);

            // Constructor from quaternion and translation vector
            EuclideanTransform(const glm::quat& rot, const glm::vec3& trans);

            // Constructor from 4x4 transformation matrix
            explicit EuclideanTransform(const glm::mat4& matrix);

            // Convert to 4x4 transformation matrix
            glm::mat4 toMat4() const;

            // Composition operator - combines two transformations
            EuclideanTransform operator*(const EuclideanTransform& other) const;

            // Compound assignment operator
            EuclideanTransform& operator*=(const EuclideanTransform& other);

            // Inverse transformation
            EuclideanTransform inv() const;

            // Check if transform is identity within epsilon tolerance
            bool isIdentity(float eps = 1e-6f) const;

            // Getters
            const glm::quat& getRotation() const { return m_rotation; }
            const glm::vec3& getTranslation() const { return m_translation; }

            // Setters
            void setRotation(const glm::quat& rot) { m_rotation = rot; }
            void setTranslation(const glm::vec3& trans) { m_translation = trans; }

            // Get Euler angles (in radians) in ZYX order
            glm::vec3 getEulerAngles() const { return glm::eulerAngles(m_rotation); }

            // Apply transformation to a point
            glm::vec3 transformPoint(const glm::vec3& point) const;

            // Apply only rotation to a vector (no translation)
            glm::vec3 transformVector(const glm::vec3& vector) const;

            glm::mat3 getRotationMat() const;

        private:
            // Orthonormalize rotation matrix
            static glm::mat4 OrthonormalizeRotation(const glm::mat4& matrix);
        };
    } // namespace geometry
} // namespace gs