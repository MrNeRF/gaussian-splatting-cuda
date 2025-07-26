
// ============================================================================
// BoundingBox.hpp - Header file for bounding box functionality
// ============================================================================

#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <torch/torch.h>
#include <cfloat> // For FLT_MAX
#include <iostream> // For error output
#include "visualizer/shader.hpp" // Your existing shader class

namespace gs {
    class BoundingBox {
    public:
        BoundingBox();
        ~BoundingBox();

        // Initialize OpenGL resources
        void init();

        // Set the bounding box from min/max points
        void setBounds(const glm::vec3& min, const glm::vec3& max);

        // Set custom transform matrix for the bounding box
        void setTransform(const glm::mat4& transform);

        // Enable/disable bounding box rendering
        void setVisible(bool visible) { visible_ = visible; }
        bool isVisible() const { return visible_; }

        // Set bounding box color
        void setColor(const glm::vec3& color) { color_ = color; }

        // Set line width
        void setLineWidth(float width) { line_width_ = width; }

        // Render the bounding box
        void render(const glm::mat4& view, const glm::mat4& projection);

        // Update bounds from model data
        void updateFromModel(const torch::Tensor& positions);

        // Auto-fit to scene
        void autoFit(const torch::Tensor& positions, float padding = 0.1f);

        // Get current bounds
        glm::vec3 getMinBounds() const { return min_bounds_; }
        glm::vec3 getMaxBounds() const { return max_bounds_; }
        glm::vec3 getCenter() const { return (min_bounds_ + max_bounds_) * 0.5f; }
        glm::vec3 getSize() const { return max_bounds_ - min_bounds_; }
        bool isInitilized() const {return  initialized_;}

    private:
        void createCubeGeometry();
        void setupVertexData();
        void cleanup();

        // OpenGL resources
        std::unique_ptr<Shader> shader_;
        GLuint VAO_, VBO_, EBO_;

        // Bounding box properties
        glm::vec3 min_bounds_;
        glm::vec3 max_bounds_;
        glm::mat4 transform_;
        glm::vec3 color_;
        float line_width_;
        bool visible_;
        bool initialized_;

        // Cube geometry data
        std::vector<glm::vec3> vertices_;
        std::vector<unsigned int> indices_;

        // Line indices for wireframe cube (12 edges, 24 indices)
        static const unsigned int cube_line_indices_[24];
    };
}