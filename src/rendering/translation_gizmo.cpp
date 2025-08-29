/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "translation_gizmo.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include "shader_paths.hpp"
#include <numbers>
#include <vector>

namespace gs::rendering {

    constexpr glm::vec3 TranslationGizmo::colors_[];

    Result<void> TranslationGizmo::initialize() {
        if (initialized_)
            return {};

        LOG_TIMER("TranslationGizmo::initialize");

        // Load shader
        auto result = load_shader("translation_gizmo", "translation_gizmo.vert",
                                  "translation_gizmo.frag", false);
        if (!result) {
            LOG_ERROR("Failed to load translation gizmo shader: {}", result.error().what());
            return std::unexpected(result.error().what());
        }
        shader_ = std::move(*result);

        if (auto arrow_result = createArrowGeometry(); !arrow_result) {
            return arrow_result;
        }

        if (auto plane_result = createPlaneGeometry(); !plane_result) {
            return plane_result;
        }

        initialized_ = true;
        LOG_INFO("TranslationGizmo initialized successfully");
        return {};
    }

    void TranslationGizmo::shutdown() {
        LOG_DEBUG("Shutting down TranslationGizmo");
        arrow_vao_ = VAO();
        arrow_vbo_ = VBO();
        plane_vao_ = VAO();
        plane_vbo_ = VBO();
        shader_ = ManagedShader();
        initialized_ = false;
    }

    Result<void> TranslationGizmo::createArrowGeometry() {
        LOG_TIMER_TRACE("TranslationGizmo::createArrowGeometry");

        std::vector<float> vertices;

        // Create cylinder shaft
        constexpr int segments = 16;
        constexpr float radius = 0.04f;
        constexpr float height = 0.4f;

        for (int i = 0; i < segments; i++) {
            float angle1 = static_cast<float>(i) * 2.0f * std::numbers::pi_v<float> / segments;
            float angle2 = static_cast<float>(i + 1) * 2.0f * std::numbers::pi_v<float> / segments;

            float x1 = radius * cos(angle1), z1 = radius * sin(angle1);
            float x2 = radius * cos(angle2), z2 = radius * sin(angle2);

            // Bottom triangle
            vertices.insert(vertices.end(), {x1, 0.0f, z1, x1, 0.0f, z1});
            vertices.insert(vertices.end(), {x2, 0.0f, z2, x2, 0.0f, z2});
            vertices.insert(vertices.end(), {x2, height, z2, x2, 0.0f, z2});

            // Top triangle
            vertices.insert(vertices.end(), {x1, 0.0f, z1, x1, 0.0f, z1});
            vertices.insert(vertices.end(), {x2, height, z2, x2, 0.0f, z2});
            vertices.insert(vertices.end(), {x1, height, z1, x1, 0.0f, z1});
        }

        // Create cone head
        constexpr float cone_radius = 0.08f;
        constexpr float cone_height = 0.15f;
        float cone_base = height;

        for (int i = 0; i < segments; i++) {
            float angle1 = static_cast<float>(i) * 2.0f * std::numbers::pi_v<float> / segments;
            float angle2 = static_cast<float>(i + 1) * 2.0f * std::numbers::pi_v<float> / segments;

            float x1 = cone_radius * cos(angle1), z1 = cone_radius * sin(angle1);
            float x2 = cone_radius * cos(angle2), z2 = cone_radius * sin(angle2);

            // Cone triangles
            vertices.insert(vertices.end(), {0.0f, cone_base + cone_height, 0.0f, 0.0f, 1.0f, 0.0f});
            vertices.insert(vertices.end(), {x1, cone_base, z1, x1, 0.0f, z1});
            vertices.insert(vertices.end(), {x2, cone_base, z2, x2, 0.0f, z2});
        }

        arrow_vertex_count_ = vertices.size() / 6;

        auto vao_result = create_vao();
        if (!vao_result)
            return std::unexpected(vao_result.error());

        auto vbo_result = create_vbo();
        if (!vbo_result)
            return std::unexpected(vbo_result.error());
        arrow_vbo_ = std::move(*vbo_result);

        VAOBuilder builder(std::move(*vao_result));
        builder.attachVBO(arrow_vbo_, std::span(vertices), GL_STATIC_DRAW)
            .setAttribute({.index = 0, .size = 3, .type = GL_FLOAT, .stride = 6 * sizeof(float), .offset = nullptr})
            .setAttribute({.index = 1, .size = 3, .type = GL_FLOAT, .stride = 6 * sizeof(float), .offset = (void*)(3 * sizeof(float))});

        arrow_vao_ = builder.build();
        LOG_DEBUG("Created arrow geometry with {} vertices", arrow_vertex_count_);
        return {};
    }

    Result<void> TranslationGizmo::createPlaneGeometry() {
        LOG_TIMER_TRACE("TranslationGizmo::createPlaneGeometry");

        float vertices[] = {
            // Positions          // Normals
            -0.5f,
            -0.5f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.5f,
            -0.5f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.5f,
            0.5f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.5f,
            0.5f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            -0.5f,
            0.5f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            -0.5f,
            -0.5f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
        };

        plane_vertex_count_ = 6;

        auto vao_result = create_vao();
        if (!vao_result)
            return std::unexpected(vao_result.error());

        auto vbo_result = create_vbo();
        if (!vbo_result)
            return std::unexpected(vbo_result.error());
        plane_vbo_ = std::move(*vbo_result);

        VAOBuilder builder(std::move(*vao_result));
        builder.attachVBO(plane_vbo_, std::span(vertices, sizeof(vertices) / sizeof(float)), GL_STATIC_DRAW)
            .setAttribute({.index = 0, .size = 3, .type = GL_FLOAT, .stride = 6 * sizeof(float), .offset = nullptr})
            .setAttribute({.index = 1, .size = 3, .type = GL_FLOAT, .stride = 6 * sizeof(float), .offset = (void*)(3 * sizeof(float))});

        plane_vao_ = builder.build();
        LOG_DEBUG("Created plane geometry");
        return {};
    }

    Result<void> TranslationGizmo::render(const glm::mat4& view, const glm::mat4& projection,
                                          const glm::vec3& position, float scale) {
        if (!initialized_) {
            LOG_ERROR("TranslationGizmo not initialized");
            return std::unexpected("TranslationGizmo not initialized");
        }

        LOG_TIMER_TRACE("TranslationGizmo::render");

        GLStateGuard state_guard;

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        ShaderScope s(shader_);

        if (auto result = s->set("u_view", view); !result)
            return result;
        if (auto result = s->set("u_projection", projection); !result)
            return result;
        if (auto result = s->set("u_light_pos", glm::vec3(10.0f, 10.0f, 10.0f)); !result)
            return result;

        // Draw planes first (behind arrows)
        constexpr float plane_size = 0.25f;
        constexpr float plane_offset = 0.15f;

        // XY Plane (Blue)
        {
            glm::mat4 model(1.0f);
            model = glm::translate(model, position + glm::vec3(plane_offset, plane_offset, 0.0f));
            model = glm::scale(model, glm::vec3(plane_size * scale));

            if (auto result = s->set("u_model", model); !result)
                return result;

            float alpha = (hovered_ == Element::XYPlane || drag_state_.element == Element::XYPlane) ? 0.8f : 0.4f;
            if (auto result = s->set("u_alpha", alpha); !result)
                return result;
            if (auto result = s->set("u_color", glm::vec3(0.0f, 0.0f, 1.0f)); !result)
                return result;

            VAOBinder vao_bind(plane_vao_);
            glDrawArrays(GL_TRIANGLES, 0, plane_vertex_count_);
        }

        // XZ Plane (Green)
        {
            glm::mat4 model(1.0f);
            model = glm::translate(model, position + glm::vec3(plane_offset, 0.0f, plane_offset));
            model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            model = glm::scale(model, glm::vec3(plane_size * scale));

            if (auto result = s->set("u_model", model); !result)
                return result;

            float alpha = (hovered_ == Element::XZPlane || drag_state_.element == Element::XZPlane) ? 0.8f : 0.4f;
            if (auto result = s->set("u_alpha", alpha); !result)
                return result;
            if (auto result = s->set("u_color", glm::vec3(0.0f, 1.0f, 0.0f)); !result)
                return result;

            VAOBinder vao_bind(plane_vao_);
            glDrawArrays(GL_TRIANGLES, 0, plane_vertex_count_);
        }

        // YZ Plane (Red)
        {
            glm::mat4 model(1.0f);
            model = glm::translate(model, position + glm::vec3(0.0f, plane_offset, plane_offset));
            model = glm::rotate(model, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::scale(model, glm::vec3(plane_size * scale));

            if (auto result = s->set("u_model", model); !result)
                return result;

            float alpha = (hovered_ == Element::YZPlane || drag_state_.element == Element::YZPlane) ? 0.8f : 0.4f;
            if (auto result = s->set("u_alpha", alpha); !result)
                return result;
            if (auto result = s->set("u_color", glm::vec3(1.0f, 0.0f, 0.0f)); !result)
                return result;

            VAOBinder vao_bind(plane_vao_);
            glDrawArrays(GL_TRIANGLES, 0, plane_vertex_count_);
        }

        // Draw arrows
        if (auto result = s->set("u_alpha", 1.0f); !result)
            return result;

        // X-axis (Red)
        {
            glm::mat4 model(1.0f);
            model = glm::translate(model, position);
            model = glm::rotate(model, glm::radians(-90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            model = glm::scale(model, glm::vec3(scale));

            if (auto result = s->set("u_model", model); !result)
                return result;

            glm::vec3 color = (hovered_ == Element::XAxis || drag_state_.element == Element::XAxis)
                                  ? glm::vec3(1.0f, 0.5f, 0.5f)
                                  : colors_[0];
            if (auto result = s->set("u_color", color); !result)
                return result;

            VAOBinder vao_bind(arrow_vao_);
            glDrawArrays(GL_TRIANGLES, 0, arrow_vertex_count_);
        }

        // Y-axis (Green)
        {
            glm::mat4 model(1.0f);
            model = glm::translate(model, position);
            model = glm::scale(model, glm::vec3(scale));

            if (auto result = s->set("u_model", model); !result)
                return result;

            glm::vec3 color = (hovered_ == Element::YAxis || drag_state_.element == Element::YAxis)
                                  ? glm::vec3(0.5f, 1.0f, 0.5f)
                                  : colors_[1];
            if (auto result = s->set("u_color", color); !result)
                return result;

            VAOBinder vao_bind(arrow_vao_);
            glDrawArrays(GL_TRIANGLES, 0, arrow_vertex_count_);
        }

        // Z-axis (Blue)
        {
            glm::mat4 model(1.0f);
            model = glm::translate(model, position);
            model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            model = glm::scale(model, glm::vec3(scale));

            if (auto result = s->set("u_model", model); !result)
                return result;

            glm::vec3 color = (hovered_ == Element::ZAxis || drag_state_.element == Element::ZAxis)
                                  ? glm::vec3(0.5f, 0.5f, 1.0f)
                                  : colors_[2];
            if (auto result = s->set("u_color", color); !result)
                return result;

            VAOBinder vao_bind(arrow_vao_);
            glDrawArrays(GL_TRIANGLES, 0, arrow_vertex_count_);
        }

        return {};
    }

    TranslationGizmo::Element TranslationGizmo::pick(const glm::vec2& mouse_pos, const glm::mat4& view,
                                                     const glm::mat4& projection, const glm::vec3& position) {
        glm::vec3 ray_origin = glm::vec3(glm::inverse(view)[3]);
        glm::vec3 ray_dir = getRayFromMouse(mouse_pos, view, projection);

        // Check planes first
        constexpr float plane_size = 0.25f;
        constexpr float plane_offset = 0.15f;

        // XY Plane
        glm::vec3 xy_center = position + glm::vec3(plane_offset, plane_offset, 0.0f);
        glm::vec3 intersection = getPlaneIntersection(ray_origin, ray_dir, glm::vec3(0, 0, 1), xy_center);
        glm::vec3 local = intersection - xy_center;
        if (std::abs(local.x) < plane_size * 0.5f && std::abs(local.y) < plane_size * 0.5f) {
            return Element::XYPlane;
        }

        // XZ Plane
        glm::vec3 xz_center = position + glm::vec3(plane_offset, 0.0f, plane_offset);
        intersection = getPlaneIntersection(ray_origin, ray_dir, glm::vec3(0, 1, 0), xz_center);
        local = intersection - xz_center;
        if (std::abs(local.x) < plane_size * 0.5f && std::abs(local.z) < plane_size * 0.5f) {
            return Element::XZPlane;
        }

        // YZ Plane
        glm::vec3 yz_center = position + glm::vec3(0.0f, plane_offset, plane_offset);
        intersection = getPlaneIntersection(ray_origin, ray_dir, glm::vec3(1, 0, 0), yz_center);
        local = intersection - yz_center;
        if (std::abs(local.y) < plane_size * 0.5f && std::abs(local.z) < plane_size * 0.5f) {
            return Element::YZPlane;
        }

        // Check axes
        float axis_threshold = 0.2f;
        float best_dist = axis_threshold;
        Element best_element = Element::None;

        auto check_axis = [&](const glm::vec3& axis_dir, Element element) {
            glm::vec3 axis_point = position + axis_dir * 0.3f;
            float dist = glm::length(glm::cross(ray_dir, axis_point - ray_origin)) / glm::length(ray_dir);
            if (dist < best_dist) {
                best_dist = dist;
                best_element = element;
            }
        };

        check_axis(glm::vec3(1, 0, 0), Element::XAxis);
        check_axis(glm::vec3(0, 1, 0), Element::YAxis);
        check_axis(glm::vec3(0, 0, 1), Element::ZAxis);

        return best_element;
    }

    glm::vec3 TranslationGizmo::startDrag(Element element, const glm::vec2& mouse_pos,
                                          const glm::mat4& view, const glm::mat4& projection,
                                          const glm::vec3& position) {
        drag_state_.active = true;
        drag_state_.element = element;
        drag_state_.start_position = position;

        glm::vec3 ray_origin = glm::vec3(glm::inverse(view)[3]);
        glm::vec3 ray_dir = getRayFromMouse(mouse_pos, view, projection);

        switch (element) {
        case Element::XAxis:
            drag_state_.axis = glm::vec3(1, 0, 0);
            drag_state_.plane_normal = glm::normalize(glm::cross(drag_state_.axis,
                                                                 glm::vec3(0, 1, 0) - glm::dot(glm::vec3(0, 1, 0), drag_state_.axis) * drag_state_.axis));
            break;
        case Element::YAxis:
            drag_state_.axis = glm::vec3(0, 1, 0);
            drag_state_.plane_normal = glm::normalize(glm::cross(drag_state_.axis,
                                                                 glm::vec3(1, 0, 0) - glm::dot(glm::vec3(1, 0, 0), drag_state_.axis) * drag_state_.axis));
            break;
        case Element::ZAxis:
            drag_state_.axis = glm::vec3(0, 0, 1);
            drag_state_.plane_normal = glm::normalize(glm::cross(drag_state_.axis,
                                                                 glm::vec3(0, 1, 0) - glm::dot(glm::vec3(0, 1, 0), drag_state_.axis) * drag_state_.axis));
            break;
        case Element::XYPlane:
            drag_state_.plane_normal = glm::vec3(0, 0, 1);
            break;
        case Element::XZPlane:
            drag_state_.plane_normal = glm::vec3(0, 1, 0);
            break;
        case Element::YZPlane:
            drag_state_.plane_normal = glm::vec3(1, 0, 0);
            break;
        default:
            break;
        }

        drag_state_.start_world_pos = getPlaneIntersection(ray_origin, ray_dir,
                                                           drag_state_.plane_normal, position);
        return position;
    }

    glm::vec3 TranslationGizmo::updateDrag(const glm::vec2& mouse_pos, const glm::mat4& view,
                                           const glm::mat4& projection) {
        if (!drag_state_.active)
            return drag_state_.start_position;

        glm::vec3 ray_origin = glm::vec3(glm::inverse(view)[3]);
        glm::vec3 ray_dir = getRayFromMouse(mouse_pos, view, projection);

        glm::vec3 intersection = getPlaneIntersection(ray_origin, ray_dir,
                                                      drag_state_.plane_normal, drag_state_.start_position);
        glm::vec3 delta = intersection - drag_state_.start_world_pos;

        if (drag_state_.element == Element::XAxis ||
            drag_state_.element == Element::YAxis ||
            drag_state_.element == Element::ZAxis) {
            float projection = glm::dot(delta, drag_state_.axis);
            return drag_state_.start_position + drag_state_.axis * projection;
        } else {
            return drag_state_.start_position + delta;
        }
    }

    glm::vec3 TranslationGizmo::getRayFromMouse(const glm::vec2& mouse, const glm::mat4& view,
                                                const glm::mat4& projection) const {
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);

        float x = (2.0f * mouse.x) / viewport[2] - 1.0f;
        float y = 1.0f - (2.0f * mouse.y) / viewport[3];

        glm::vec4 ray_clip(x, y, -1.0f, 1.0f);
        glm::vec4 ray_eye = glm::inverse(projection) * ray_clip;
        ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0f, 0.0f);

        return glm::normalize(glm::vec3(glm::inverse(view) * ray_eye));
    }

    glm::vec3 TranslationGizmo::getPlaneIntersection(const glm::vec3& ray_origin, const glm::vec3& ray_dir,
                                                     const glm::vec3& plane_normal, const glm::vec3& plane_point) const {
        float denominator = glm::dot(plane_normal, ray_dir);
        if (std::abs(denominator) > 0.0001f) {
            float t = glm::dot(plane_point - ray_origin, plane_normal) / denominator;
            if (t >= 0) {
                return ray_origin + t * ray_dir;
            }
        }
        return plane_point;
    }

} // namespace gs::rendering
