#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <memory>

namespace gs::rendering {

    class Shader;
    class TextRenderer;

    class ViewportGizmo {
    public:
        ViewportGizmo();
        ~ViewportGizmo();

        // Initialize OpenGL resources
        void initialize();

        // Render the gizmo
        void render(const glm::mat3& camera_rotation,
                    const glm::vec2& viewport_pos,
                    const glm::vec2& viewport_size);

        // Cleanup
        void shutdown();

        // Settings
        void setSize(int size) { size_ = size; }
        void setMargin(int margin) { margin_ = margin; }
        int getSize() const { return size_; }
        int getMargin() const { return margin_; }

    private:
        void generateGeometry();
        void createShaders();

        // OpenGL resources
        GLuint vao_ = 0;
        GLuint vbo_ = 0;
        std::unique_ptr<Shader> shader_;

        // Text rendering
        std::unique_ptr<TextRenderer> text_renderer_;

        // Geometry info
        int cylinder_vertex_count_ = 0;
        int sphere_vertex_start_ = 0;
        int sphere_vertex_count_ = 0;

        // Settings
        int size_ = 95;
        int margin_ = 10;
        bool initialized_ = false;

        // Gizmo colors (RGB -> XYZ)
        static constexpr glm::vec3 colors_[3] = {
            {0.89f, 0.15f, 0.21f}, // X - Red
            {0.54f, 0.86f, 0.20f}, // Y - Green
            {0.17f, 0.48f, 0.87f}  // Z - Blue
        };
    };

} // namespace gs::rendering