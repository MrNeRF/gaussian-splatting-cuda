#include "geometry/bounding_box.hpp"

#include <stdexcept>
// ============================================================================
// BoundingBox.cpp - Implementation
// ============================================================================
namespace gs {
    namespace geometry {

        BoundingBox::BoundingBox()
            : min_bounds_(-1.0f, -1.0f, -1.0f),
              max_bounds_(1.0f, 1.0f, 1.0f),
              world2BBox_(EuclideanTransform()) {}

        BoundingBox::~BoundingBox() {}

        void BoundingBox::setBounds(const glm::vec3& min, const glm::vec3& max) {
            // Validate bounds
            if (min.x > max.x || min.y > max.y || min.z > max.z) {
                throw std::runtime_error("Warning: Invalid bounding box bounds (min > max)");
            }

            min_bounds_ = min;
            max_bounds_ = max;
        }

        void BoundingBox::setworld2BBox(const geometry::EuclideanTransform& transform) {
            world2BBox_ = transform;
        }

        glm::vec3 BoundingBox::getCenter() const {
            const auto local_center = (min_bounds_ + max_bounds_) * 0.5f;
            const auto world_center = world2BBox_.inv().transformPoint(local_center);
            return glm::vec3{world_center};
        }

        glm::vec3 BoundingBox::getLocalCenter() const {
            const auto local_center = (min_bounds_ + max_bounds_) * 0.5f;
            return glm::vec3{local_center};
        }

    } // namespace geometry
} // namespace gs