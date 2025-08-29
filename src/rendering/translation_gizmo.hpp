/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <functional>
#include <glm/glm.hpp>

namespace gs::rendering {

    class TranslationGizmo {
    public:
        enum class Element {
            None,
            XAxis,
            YAxis,
            ZAxis,
            XYPlane,
            XZPlane,
            YZPlane
        };

        struct DragState {
            bool active = false;
            Element element = Element::None;
            glm::vec3 start_position;
            glm::vec3 start_world_pos;
            glm::vec3 plane_normal;
            glm::vec3 axis;
        };

        TranslationGizmo() = default;
        ~TranslationGizmo() = default;

        Result<void> initialize();
        void shutdown();

        Result<void> render(const glm::mat4& view, const glm::mat4& projection,
                            const glm::vec3& position, float scale = 1.0f);

        // Interaction
        Element pick(const glm::vec2& mouse_pos, const glm::mat4& view,
                     const glm::mat4& projection, const glm::vec3& position);

        glm::vec3 startDrag(Element element, const glm::vec2& mouse_pos,
                            const glm::mat4& view, const glm::mat4& projection,
                            const glm::vec3& position);

        glm::vec3 updateDrag(const glm::vec2& mouse_pos, const glm::mat4& view,
                             const glm::mat4& projection);

        void endDrag() { drag_state_.active = false; }
        bool isDragging() const { return drag_state_.active; }

        // Visual state
        void setHoveredElement(Element elem) { hovered_ = elem; }
        Element getHoveredElement() const { return hovered_; }

    private:
        Result<void> createArrowGeometry();
        Result<void> createPlaneGeometry();

        glm::vec3 getRayFromMouse(const glm::vec2& mouse, const glm::mat4& view,
                                  const glm::mat4& projection) const;
        glm::vec3 getPlaneIntersection(const glm::vec3& ray_origin, const glm::vec3& ray_dir,
                                       const glm::vec3& plane_normal, const glm::vec3& plane_point) const;

        // Resources
        ManagedShader shader_;
        VAO arrow_vao_, plane_vao_;
        VBO arrow_vbo_, plane_vbo_;

        // Geometry info
        size_t arrow_vertex_count_ = 0;
        size_t plane_vertex_count_ = 0;

        // State
        Element hovered_ = Element::None;
        DragState drag_state_;
        bool initialized_ = false;

        // Colors
        static constexpr glm::vec3 colors_[3] = {
            {1.0f, 0.2f, 0.2f}, // X - Red
            {0.2f, 1.0f, 0.2f}, // Y - Green
            {0.2f, 0.2f, 1.0f}  // Z - Blue
        };
    };

} // namespace gs::rendering
