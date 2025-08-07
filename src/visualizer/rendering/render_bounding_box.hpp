#pragma once

#include "geometry/bounding_box.hpp"

#include "rendering/shader.hpp"
#include <memory>

namespace gs {
    class RenderBoundingBox : public geometry::BoundingBox {
    public:
        RenderBoundingBox();
        ~RenderBoundingBox();

        // Set the bounding box from min/max points
        void setBounds(const glm::vec3& min, const glm::vec3& max) override;

        // Initialize OpenGL resources
        void init();

        // Enable/disable bounding box rendering
        // void setVisible(bool visible) { visible_ = visible; }
        // bool isVisible() const { return visible_; }

        bool isInitialized() const { return initialized_; }

        // Set bounding box color
        void setColor(const glm::vec3& color) { color_ = color; }

        // Set line width
        void setLineWidth(float width) { line_width_ = width; }

        // Render the bounding box
        void render(const glm::mat4& view, const glm::mat4& projection);

        bool isInitilized() const { return initialized_; }

    private:
        void createCubeGeometry();
        void setupVertexData();
        void cleanup();

        // Bounding box properties
        glm::vec3 color_;
        float line_width_;
        bool initialized_;

        // OpenGL resources
        std::unique_ptr<Shader> shader_;
        GLuint VAO_, VBO_, EBO_;

        // Cube geometry data
        std::vector<glm::vec3> vertices_;
        std::vector<unsigned int> indices_;

        // Line indices for wireframe cube (12 edges, 24 indices)
        static const unsigned int cube_line_indices_[24];
    };
} // namespace gs