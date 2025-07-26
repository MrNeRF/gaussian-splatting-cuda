#include "core/bounding_box.hpp"

#include <stdexcept>
// ============================================================================
    // BoundingBox.cpp - Implementation
    // ============================================================================
namespace gs {


    BoundingBox::BoundingBox()
        : min_bounds_(-1.0f, -1.0f, -1.0f)
        , max_bounds_(1.0f, 1.0f, 1.0f)
        , transform_(1.0f)
    {}

    BoundingBox::~BoundingBox() {}


    void BoundingBox::setBounds(const glm::vec3& min, const glm::vec3& max) {
        // Validate bounds
        if (min.x > max.x || min.y > max.y || min.z > max.z) {
           throw std::runtime_error("Warning: Invalid bounding box bounds (min > max)");
        }

        min_bounds_ = min;
        max_bounds_ = max;
    }

    void BoundingBox::setTransform(const glm::mat4& transform) {
        transform_ = transform;
    }

}