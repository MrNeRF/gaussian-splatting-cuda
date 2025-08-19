#pragma once

#include "geometry/bounding_box.hpp"
#include "rendering/rendering.hpp"
#include "shader_manager.hpp"

namespace gs::rendering {
    class RenderBoundingBox : public geometry::BoundingBox, public IBoundingBox {
    public:
        RenderBoundingBox();
        ~RenderBoundingBox() override;

        // Set the bounding box from min/max points
        void setBounds(const glm::vec3& min, const glm::vec3& max) override;

        // Initialize OpenGL resources
        void init();

        // Check if initialized
        bool isInitialized() const override { return initialized_; }

        // IBoundingBox interface implementation
        glm::vec3 getMinBounds() const override { return min_bounds_; }
        glm::vec3 getMaxBounds() const override { return max_bounds_; }
        glm::vec3 getCenter() const override { return BoundingBox::getCenter(); }
        glm::vec3 getSize() const override { return BoundingBox::getSize(); }
        glm::vec3 getLocalCenter() const override { return BoundingBox::getLocalCenter(); }

        void setworld2BBox(const geometry::EuclideanTransform& transform) override {
            BoundingBox::setworld2BBox(transform);
        }
        geometry::EuclideanTransform getworld2BBox() const override {
            return BoundingBox::getworld2BBox();
        }

        // Set bounding box color
        void setColor(const glm::vec3& color) override { color_ = color; }

        // Set line width
        void setLineWidth(float width) override { line_width_ = width; }

        // Get color and line width
        glm::vec3 getColor() const override { return color_; }
        float getLineWidth() const override { return line_width_; }

        // Render the bounding box
        void render(const glm::mat4& view, const glm::mat4& projection);

    private:
        void createCubeGeometry();
        void setupVertexData();
        void cleanup();

        // Bounding box properties
        glm::vec3 color_;
        float line_width_;
        bool initialized_;

        // OpenGL resources
        ManagedShader shader_;
        GLuint VAO_, VBO_, EBO_;

        // Cube geometry data
        std::vector<glm::vec3> vertices_;
        std::vector<unsigned int> indices_;

        // Line indices for wireframe cube (12 edges, 24 indices)
        static const unsigned int cube_line_indices_[24];
    };
} // namespace gs::rendering
