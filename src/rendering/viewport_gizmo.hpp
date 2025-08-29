/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>
#include <memory>

namespace gs::rendering {

    class TextRenderer; // Forward declaration

    class ViewportGizmo {
    public:
        ViewportGizmo();  // Declare constructor (not defaulted)
        ~ViewportGizmo(); // Declare destructor

        // Initialize OpenGL resources - now returns Result
        Result<void> initialize();

        // Render the gizmo - now returns Result
        Result<void> render(const glm::mat3& camera_rotation,
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
        Result<void> generateGeometry();
        Result<void> createShaders();

        // OpenGL resources using RAII
        VAO vao_;
        VBO vbo_;
        ManagedShader shader_;

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
