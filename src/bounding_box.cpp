#include "core/bounding_box.hpp"

#include <stdexcept>
// ============================================================================
// BoundingBox.cpp - Implementation
// ============================================================================
namespace gs {

    BoundingBox::BoundingBox()
        : min_bounds_(-1.0f, -1.0f, -1.0f),
          max_bounds_(1.0f, 1.0f, 1.0f),
          world2BBox_(1.0f) {}

    BoundingBox::~BoundingBox() {}

    void BoundingBox::setBounds(const glm::vec3& min, const glm::vec3& max) {
        // Validate bounds
        if (min.x > max.x || min.y > max.y || min.z > max.z) {
            throw std::runtime_error("Warning: Invalid bounding box bounds (min > max)");
        }

        min_bounds_ = min;
        max_bounds_ = max;
    }

    void BoundingBox::setworld2BBox(const glm::mat4& transform) {
        world2BBox_ = transform;
    }

    glm::vec3 BoundingBox::getCenter() const {
        const auto local_center = (min_bounds_ + max_bounds_) * 0.5f;
        const auto world_center = glm::inverse(world2BBox_) * glm::vec4{local_center, 1.0f};
        return glm::vec3{world_center};
    }

    glm::vec3 BoundingBox::getLocalCenter() const {
        const auto local_center = (min_bounds_ + max_bounds_) * 0.5f;
        return glm::vec3{local_center};
    }

} // namespace gs