#pragma once

#include "shader.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace gs::rendering {
    class RenderCoordinateAxes {
    public:
        RenderCoordinateAxes();
        ~RenderCoordinateAxes();

        // Set the size (length) of the axes
        void setSize(float size);
        float getSize() const { return size_; }

        // Initialize OpenGL resources
        void init();

        // Check if initialized
        bool isInitialized() const { return initialized_; }

        // Set line width for axes
        void setLineWidth(float width) { line_width_ = width; }

        // Enable/disable individual axes
        void setAxisVisible(int axis, bool visible); // 0=X, 1=Y, 2=Z
        bool isAxisVisible(int axis) const;

        // Render the coordinate axes
        void render(const glm::mat4& view, const glm::mat4& projection);

    private:
        void createAxesGeometry();
        void setupVertexData();
        void cleanup();

        // OpenGL resources
        std::unique_ptr<Shader> shader_;
        GLuint VAO_, VBO_;

        // Axes properties
        float size_;
        float line_width_;
        bool initialized_;
        bool axis_visible_[3]; // X, Y, Z visibility

        // Standard colors for coordinate axes
        static const glm::vec3 X_AXIS_COLOR; // Red
        static const glm::vec3 Y_AXIS_COLOR; // Green
        static const glm::vec3 Z_AXIS_COLOR; // Blue

        // Geometry data - vertices with colors
        struct AxisVertex {
            glm::vec3 position;
            glm::vec3 color;
        };

        std::vector<AxisVertex> vertices_;
    };
} // namespace gs::rendering