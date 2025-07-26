
// ============================================================================
// BoundingBox.hpp - Header file for bounding box functionality
// ============================================================================

#pragma once

#include <glm/glm.hpp>

namespace gs {
    class BoundingBox {
    public:
        BoundingBox();
        virtual ~BoundingBox();

        // Set the bounding box from min/max points
        virtual void setBounds(const glm::vec3& min, const glm::vec3& max);

        // Set custom transform matrix for the bounding box
        void setworld2BBox(const glm::mat4& transform);

        // Get current bounds
        glm::vec3 getMinBounds() const { return min_bounds_; }
        glm::vec3 getMaxBounds() const { return max_bounds_; }
        glm::vec3 getCenter() const { return (min_bounds_ + max_bounds_) * 0.5f; }
        glm::vec3 getSize() const { return max_bounds_ - min_bounds_; }

    protected:
        // Bounding box properties
        glm::vec3 min_bounds_;
        glm::vec3 max_bounds_;
        // relative position of boundin box to world
        glm::mat4 world2BBox_;
    };
} // namespace gs