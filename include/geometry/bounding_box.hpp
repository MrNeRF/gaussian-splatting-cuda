/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */


#pragma once

#include <glm/glm.hpp>

#include "geometry/euclidean_transform.hpp"

namespace gs {
    namespace geometry {
        class BoundingBox {
        public:
            BoundingBox();
            virtual ~BoundingBox();

            // Set the bounding box from min/max points
            virtual void setBounds(const glm::vec3& min, const glm::vec3& max);

            // Set custom transform matrix for the bounding box
            void setworld2BBox(const geometry::EuclideanTransform& transform);
            // getter
            const geometry::EuclideanTransform& getworld2BBox() const { return world2BBox_; }

            // Get current bounds
            glm::vec3 getMinBounds() const { return min_bounds_; }
            glm::vec3 getMaxBounds() const { return max_bounds_; }
            glm::vec3 getCenter() const;
            glm::vec3 getLocalCenter() const;
            glm::vec3 getSize() const { return max_bounds_ - min_bounds_; }

        protected:
            // Bounding box properties
            glm::vec3 min_bounds_;
            glm::vec3 max_bounds_;
            // relative position of bounding box to the world
            EuclideanTransform world2BBox_;
        };
    } // namespace geometry
} // namespace gs