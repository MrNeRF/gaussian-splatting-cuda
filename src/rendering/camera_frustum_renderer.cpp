/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "camera_frustum_renderer.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace gs::rendering {

    Result<void> CameraFrustumRenderer::init() {
        LOG_DEBUG("Initializing camera frustum renderer");

        // Load shader
        auto shader_result = load_shader("camera_frustum", "camera_frustum.vert", "camera_frustum.frag", false);
        if (!shader_result) {
            LOG_ERROR("Failed to load camera frustum shader: {}", shader_result.error().what());
            return std::unexpected(shader_result.error().what());
        }
        shader_ = std::move(*shader_result);

        // Create geometry
        if (auto result = createGeometry(); !result) {
            return result;
        }

        // Create instance buffer
        auto instance_vbo_result = create_vbo();
        if (!instance_vbo_result) {
            return std::unexpected(instance_vbo_result.error());
        }
        instance_vbo_ = std::move(*instance_vbo_result);

        initialized_ = true;
        LOG_INFO("Camera frustum renderer initialized");
        return {};
    }

    Result<void> CameraFrustumRenderer::createGeometry() {
        LOG_TIMER_TRACE("CameraFrustumRenderer::createGeometry");

        // Frustum vertices in camera space (apex at origin, base at z=-1)
        std::vector<glm::vec3> vertices = {
            // Base vertices (at z = -1, sized by FOV)
            {-0.5f, -0.5f, -1.0f}, // 0 bottom-left
            {0.5f, -0.5f, -1.0f},  // 1 bottom-right
            {0.5f, 0.5f, -1.0f},   // 2 top-right
            {-0.5f, 0.5f, -1.0f},  // 3 top-left
            // Apex (camera position)
            {0.0f, 0.0f, 0.0f} // 4
        };

        // Face indices (triangles)
        std::vector<unsigned int> face_indices = {
            // Base (facing away)
            0, 1, 2,
            0, 2, 3,
            // Side faces
            0, 4, 1,
            1, 4, 2,
            2, 4, 3,
            3, 4, 0};

        // Edge indices (lines)
        std::vector<unsigned int> edge_indices = {
            0, 1, 1, 2, 2, 3, 3, 0, // Base edges
            0, 4, 1, 4, 2, 4, 3, 4  // Apex edges
        };

        num_face_indices_ = face_indices.size();
        num_edge_indices_ = edge_indices.size();

        // Create VAO and buffers
        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected(vao_result.error());
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            return std::unexpected(vbo_result.error());
        }
        vbo_ = std::move(*vbo_result);

        auto face_ebo_result = create_vbo();
        if (!face_ebo_result) {
            return std::unexpected(face_ebo_result.error());
        }
        face_ebo_ = std::move(*face_ebo_result);

        auto edge_ebo_result = create_vbo();
        if (!edge_ebo_result) {
            return std::unexpected(edge_ebo_result.error());
        }
        edge_ebo_ = std::move(*edge_ebo_result);

        // Build VAO
        VAOBuilder builder(std::move(*vao_result));

        // Vertex positions
        std::span<const float> vertices_data(
            reinterpret_cast<const float*>(vertices.data()),
            vertices.size() * 3);

        builder.attachVBO(vbo_, vertices_data, GL_STATIC_DRAW)
            .setAttribute({.index = 0, .size = 3, .type = GL_FLOAT});

        // Face indices
        builder.attachEBO(face_ebo_, std::span(face_indices), GL_STATIC_DRAW);

        vao_ = builder.build();

        // Also upload edge indices
        BufferBinder<GL_ELEMENT_ARRAY_BUFFER> edge_bind(edge_ebo_);
        upload_buffer(GL_ELEMENT_ARRAY_BUFFER, std::span(edge_indices), GL_STATIC_DRAW);

        LOG_DEBUG("Camera frustum geometry created");
        return {};
    }

    Result<void> CameraFrustumRenderer::render(
        const std::vector<std::shared_ptr<const Camera>>& cameras,
        const glm::mat4& view,
        const glm::mat4& projection,
        float scale,
        const glm::vec3& train_color,
        const glm::vec3& eval_color) {

        if (!initialized_ || cameras.empty()) {
            return {};
        }

        LOG_DEBUG("Rendering {} camera frustums", cameras.size());

        // Prepare instance data
        std::vector<InstanceData> instances;
        instances.reserve(cameras.size());

        // Transform from OpenGL to COLMAP coordinates
        const glm::mat4 GL_TO_COLMAP = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, -1.0f));

        for (const auto& cam : cameras) {
            // Get camera world-to-camera transform
            auto R_tensor = cam->R();
            auto T_tensor = cam->T();

            if (!R_tensor.defined() || !T_tensor.defined()) {
                continue;
            }

            // Convert to CPU
            R_tensor = R_tensor.to(torch::kCPU);
            T_tensor = T_tensor.to(torch::kCPU);

            // Build world-to-camera matrix
            glm::mat4 w2c(1.0f);
            auto R_acc = R_tensor.accessor<float, 2>();
            auto T_acc = T_tensor.accessor<float, 1>();

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    w2c[j][i] = R_acc[i][j]; // Column-major
                }
                w2c[3][i] = T_acc[i];
            }

            // Camera-to-world transform
            glm::mat4 c2w = glm::inverse(w2c);

            // Apply coordinate system conversion and scale
            glm::mat4 model = c2w * GL_TO_COLMAP * glm::scale(glm::mat4(1.0f), glm::vec3(scale));

            // Determine color based on camera type (simple heuristic: test cameras often have "test" in name)
            bool is_test = cam->image_name().find("test") != std::string::npos;
            glm::vec3 color = is_test ? eval_color : train_color;

            instances.push_back({model, color, 0.0f});
        }

        if (instances.empty()) {
            return {};
        }

        // Use comprehensive state guard for entire render operation
        GLStateGuard state_guard;

        // Clear any previous OpenGL errors
        while (glGetError() != GL_NO_ERROR) {}

        // Bind shader using RAII - this scope encompasses all uniform setting and drawing
        {
            ShaderScope shader(shader_);

            if (!shader.isBound()) {
                LOG_ERROR("Failed to bind camera frustum shader");
                return std::unexpected("Failed to bind camera frustum shader");
            }

            // Set uniforms while shader is bound
            glm::mat4 view_proj = projection * view;

            // Set required uniforms (these exist in the shader)
            if (auto result = shader->set("viewProj", view_proj); !result) {
                LOG_ERROR("Failed to set viewProj uniform: {}", result.error());
            }

            if (auto result = shader->set("viewPos", glm::vec3(glm::inverse(view)[3])); !result) {
                LOG_ERROR("Failed to set viewPos uniform: {}", result.error());
            }

            // Try to set optional uniforms
            if (auto result = shader->set("highlightIndex", -1); !result) {
                LOG_TRACE("highlightIndex uniform not found (may be optimized out)");
            }

            if (auto result = shader->set("highlightColor", glm::vec3(1.0f, 1.0f, 0.0f)); !result) {
                LOG_TRACE("highlightColor uniform not found (may be optimized out)");
            }

            // Bind VAO using RAII
            {
                VAOBinder vao_bind(vao_);

                // Upload instance data using RAII buffer binding
                {
                    BufferBinder<GL_ARRAY_BUFFER> instance_bind(instance_vbo_);
                    upload_buffer(GL_ARRAY_BUFFER, std::span(instances), GL_DYNAMIC_DRAW);

                    // Setup instance attributes while buffer is bound
                    // Instance transform matrix (locations 1-4)
                    for (int i = 0; i < 4; ++i) {
                        glEnableVertexAttribArray(1 + i);
                        glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                              reinterpret_cast<void*>(sizeof(glm::vec4) * i));
                        glVertexAttribDivisor(1 + i, 1);
                    }

                    // Instance color (location 5)
                    glEnableVertexAttribArray(5);
                    glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                          reinterpret_cast<void*>(offsetof(InstanceData, color)));
                    glVertexAttribDivisor(5, 1);
                } // BufferBinder automatically unbinds here

                // Setup render state
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LESS);
                glDepthMask(GL_TRUE);
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

                // First pass: solid faces with depth
                if (auto result = shader->set("enableShading", 1); !result) {
                    LOG_ERROR("Failed to set enableShading uniform: {}", result.error());
                }

                {
                    BufferBinder<GL_ELEMENT_ARRAY_BUFFER> face_bind(face_ebo_);
                    glDrawElementsInstanced(GL_TRIANGLES, num_face_indices_, GL_UNSIGNED_INT, 0, instances.size());
                }

                // Check for errors after first draw
                GLenum err = glGetError();
                if (err != GL_NO_ERROR) {
                    LOG_ERROR("OpenGL error after drawing faces: 0x{:x}", err);
                }

                // Second pass: wireframe edges
                glLineWidth(1.0f);
                if (auto result = shader->set("enableShading", 0); !result) {
                    LOG_ERROR("Failed to set enableShading uniform for wireframe: {}", result.error());
                }

                {
                    BufferBinder<GL_ELEMENT_ARRAY_BUFFER> edge_bind(edge_ebo_);
                    glDrawElementsInstanced(GL_LINES, num_edge_indices_, GL_UNSIGNED_INT, 0, instances.size());
                }

                // Check for errors after second draw
                err = glGetError();
                if (err != GL_NO_ERROR) {
                    LOG_ERROR("OpenGL error after drawing edges: 0x{:x}", err);
                }

                // Cleanup instance attributes before VAO unbinds
                for (int i = 1; i <= 5; ++i) {
                    glDisableVertexAttribArray(i);
                    if (i >= 1 && i <= 5) {
                        glVertexAttribDivisor(i, 0);
                    }
                }
            } // VAOBinder automatically unbinds here
        } // ShaderScope automatically unbinds here

        // GLStateGuard will restore all OpenGL state when it goes out of scope

        LOG_TRACE("Rendered {} camera frustums", instances.size());
        return {};
    }

} // namespace gs::rendering