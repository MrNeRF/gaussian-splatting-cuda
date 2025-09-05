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

        // Create picking FBO
        if (auto result = createPickingFBO(); !result) {
            return result;
        }

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

    Result<void> CameraFrustumRenderer::createPickingFBO() {
        LOG_DEBUG("Creating picking framebuffer");

        // Create FBO
        GLuint fbo_id;
        glGenFramebuffers(1, &fbo_id);
        if (fbo_id == 0) {
            return std::unexpected("Failed to create picking FBO");
        }
        picking_fbo_ = FBO(fbo_id);

        // Initial size (will resize on first use)
        picking_fbo_width_ = 256;
        picking_fbo_height_ = 256;

        // Create color texture
        GLuint color_tex;
        glGenTextures(1, &color_tex);
        picking_color_texture_ = Texture(color_tex);

        glBindTexture(GL_TEXTURE_2D, picking_color_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, picking_fbo_width_, picking_fbo_height_,
                     0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Create depth texture
        GLuint depth_tex;
        glGenTextures(1, &depth_tex);
        picking_depth_texture_ = Texture(depth_tex);

        glBindTexture(GL_TEXTURE_2D, picking_depth_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, picking_fbo_width_, picking_fbo_height_,
                     0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Attach to FBO
        glBindFramebuffer(GL_FRAMEBUFFER, picking_fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, picking_color_texture_, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, picking_depth_texture_, 0);

        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("Picking FBO incomplete: 0x{:x}", status);
            return std::unexpected("Picking FBO incomplete");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        LOG_DEBUG("Picking FBO created successfully");
        return {};
    }

    void CameraFrustumRenderer::prepareInstances(const std::vector<std::shared_ptr<const Camera>>& cameras,
                                                 float scale,
                                                 const glm::vec3& train_color,
                                                 const glm::vec3& eval_color,
                                                 bool for_picking,
                                                 const glm::vec3& view_position) {

        // Track if we need to regenerate
        bool needs_regeneration = false;

        // Check if we need to regenerate instances
        if (cached_instances_.size() != cameras.size()) {
            needs_regeneration = true;
            LOG_TRACE("Instance count changed: {} -> {}", cached_instances_.size(), cameras.size());
        } else if (last_scale_ != scale || last_train_color_ != train_color || last_eval_color_ != eval_color) {
            needs_regeneration = true;
            LOG_TRACE("Instance parameters changed");
        }

        // Only regenerate if necessary
        if (!needs_regeneration && !cached_instances_.empty()) {
            // Update visibility based on distance even when using cache
            updateInstanceVisibility(view_position);
            LOG_TRACE("Using {} cached instances for {}, updating visibility",
                      cached_instances_.size(), for_picking ? "picking" : "rendering");
            return;
        }

        LOG_DEBUG("Regenerating {} instances for {} (scale: {}, train_color: [{}, {}, {}])",
                  cameras.size(), for_picking ? "picking" : "rendering", scale,
                  train_color.r, train_color.g, train_color.b);

        cached_instances_.clear();
        cached_instances_.reserve(cameras.size());
        camera_ids_.clear();
        camera_ids_.reserve(cameras.size());
        camera_positions_.clear();
        camera_positions_.reserve(cameras.size());

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

            // Extract camera position
            glm::vec3 cam_pos = glm::vec3(c2w[3]);
            camera_positions_.push_back(cam_pos);

            // Apply coordinate system conversion and scale
            glm::mat4 model = c2w * GL_TO_COLMAP * glm::scale(glm::mat4(1.0f), glm::vec3(scale));

            // Determine color based on camera type
            bool is_test = cam->image_name().find("test") != std::string::npos;
            glm::vec3 color = is_test ? eval_color : train_color;

            // Calculate alpha based on distance to view position
            float distance = glm::length(cam_pos - view_position);
            float alpha = 1.0f;

            if (for_picking) {
                // For picking, keep everything at full alpha so it's pickable
                // The picking shader will handle the actual visibility
                alpha = 1.0f;
            } else {
                // For rendering, use aggressive fading
                const float FADE_START_DISTANCE = 5.0f * scale;
                const float FADE_END_DISTANCE = 0.2f * scale;
                const float MINIMUM_VISIBLE_DISTANCE = 0.1f * scale;

                if (distance < MINIMUM_VISIBLE_DISTANCE) {
                    alpha = 0.0f; // Completely invisible when very close
                } else if (distance < FADE_END_DISTANCE) {
                    alpha = 0.05f; // Very faint
                } else if (distance < FADE_START_DISTANCE) {
                    float t = (distance - FADE_END_DISTANCE) / (FADE_START_DISTANCE - FADE_END_DISTANCE);
                    alpha = 0.05f + 0.95f * (t * t * (3.0f - 2.0f * t));
                }
            }

            cached_instances_.push_back({model, color, alpha});
            camera_ids_.push_back(cam->uid());
        }

        // Update cache parameters
        last_scale_ = scale;
        last_train_color_ = train_color;
        last_eval_color_ = eval_color;
        last_view_position_ = view_position;

        LOG_DEBUG("Prepared {} instances", cached_instances_.size());
    }

    void CameraFrustumRenderer::updateInstanceVisibility(const glm::vec3& view_position) {
        if (camera_positions_.size() != cached_instances_.size()) {
            LOG_WARN("Cannot update visibility: position count {} != instance count {}",
                     camera_positions_.size(), cached_instances_.size());
            return;
        }

        for (size_t i = 0; i < camera_positions_.size(); ++i) {
            float distance = glm::length(camera_positions_[i] - view_position);
            float alpha = 1.0f;

            // For rendering, use aggressive fading
            const float FADE_START_DISTANCE = 5.0f * last_scale_;
            const float FADE_END_DISTANCE = 0.2f * last_scale_;
            const float MINIMUM_VISIBLE_DISTANCE = 0.1f * last_scale_;

            if (distance < MINIMUM_VISIBLE_DISTANCE) {
                alpha = 0.0f;
            } else if (distance < FADE_END_DISTANCE) {
                alpha = 0.05f; // Very faint but still slightly visible
            } else if (distance < FADE_START_DISTANCE) {
                float t = (distance - FADE_END_DISTANCE) / (FADE_START_DISTANCE - FADE_END_DISTANCE);
                alpha = 0.05f + 0.95f * (t * t * (3.0f - 2.0f * t));
            }

            cached_instances_[i].alpha = alpha;
        }

        last_view_position_ = view_position;
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

        LOG_TRACE("Rendering {} camera frustums", cameras.size());

        // Extract view position from inverse of view matrix
        glm::vec3 view_position = glm::vec3(glm::inverse(view)[3]);

        // Prepare instance data for rendering (not picking)
        prepareInstances(cameras, scale, train_color, eval_color, false, view_position);

        if (cached_instances_.empty()) {
            return {};
        }

        // Filter out instances with alpha = 0 for rendering
        std::vector<InstanceData> visible_instances;
        std::vector<int> visible_indices;
        visible_instances.reserve(cached_instances_.size());
        visible_indices.reserve(cached_instances_.size());

        for (size_t i = 0; i < cached_instances_.size(); ++i) {
            if (cached_instances_[i].alpha > 0.01f) { // Skip nearly invisible frustums
                visible_instances.push_back(cached_instances_[i]);
                visible_indices.push_back(static_cast<int>(i));
            }
        }

        if (visible_instances.empty()) {
            LOG_TRACE("No visible camera frustums to render");
            return {};
        }

        LOG_TRACE("Rendering {} visible frustums out of {} total", visible_instances.size(), cached_instances_.size());

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

            // Set required uniforms
            if (auto result = shader->set("viewProj", view_proj); !result) {
                LOG_ERROR("Failed to set viewProj uniform: {}", result.error());
            }

            if (auto result = shader->set("viewPos", view_position); !result) {
                LOG_ERROR("Failed to set viewPos uniform: {}", result.error());
            }

            // Set picking mode to false for normal rendering
            if (auto result = shader->set("pickingMode", false); !result) {
                LOG_TRACE("pickingMode uniform not found");
            }

            // Find the actual highlighted index in the visible instances
            int visible_highlight_index = -1;
            if (highlighted_camera_ >= 0 && highlighted_camera_ < static_cast<int>(visible_indices.size())) {
                for (size_t i = 0; i < visible_indices.size(); ++i) {
                    if (visible_indices[i] == highlighted_camera_) {
                        visible_highlight_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            // Set highlight index
            if (auto result = shader->set("highlightIndex", visible_highlight_index); !result) {
                LOG_TRACE("highlightIndex uniform not found");
            }

            if (auto result = shader->set("highlightColor", glm::vec3(1.0f, 0.85f, 0.0f)); !result) {
                LOG_TRACE("highlightColor uniform not found");
            }

            // Bind VAO using RAII
            {
                VAOBinder vao_bind(vao_);

                // Upload visible instance data using RAII buffer binding
                {
                    BufferBinder<GL_ARRAY_BUFFER> instance_bind(instance_vbo_);
                    upload_buffer(GL_ARRAY_BUFFER, std::span(visible_instances), GL_DYNAMIC_DRAW);

                    // Setup instance attributes while buffer is bound
                    // Instance transform matrix (locations 1-4)
                    for (int i = 0; i < 4; ++i) {
                        glEnableVertexAttribArray(1 + i);
                        glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                              reinterpret_cast<void*>(sizeof(glm::vec4) * i));
                        glVertexAttribDivisor(1 + i, 1);
                    }

                    // Instance color and alpha (location 5) - now vec4
                    glEnableVertexAttribArray(5);
                    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                          reinterpret_cast<void*>(offsetof(InstanceData, color)));
                    glVertexAttribDivisor(5, 1);
                }

                // Setup render state
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LESS);
                glDepthMask(GL_TRUE);
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

                // First pass: solid faces with depth
                if (auto result = shader->set("enableShading", true); !result) {
                    LOG_ERROR("Failed to set enableShading uniform: {}", result.error());
                }

                {
                    BufferBinder<GL_ELEMENT_ARRAY_BUFFER> face_bind(face_ebo_);
                    glDrawElementsInstanced(GL_TRIANGLES, num_face_indices_, GL_UNSIGNED_INT, 0, visible_instances.size());
                }

                // Check for errors after first draw
                GLenum err = glGetError();
                if (err != GL_NO_ERROR) {
                    LOG_ERROR("OpenGL error after drawing faces: 0x{:x}", err);
                }

                // Second pass: wireframe edges
                glLineWidth(1.0f);
                if (auto result = shader->set("enableShading", false); !result) {
                    LOG_ERROR("Failed to set enableShading uniform for wireframe: {}", result.error());
                }

                {
                    BufferBinder<GL_ELEMENT_ARRAY_BUFFER> edge_bind(edge_ebo_);
                    glDrawElementsInstanced(GL_LINES, num_edge_indices_, GL_UNSIGNED_INT, 0, visible_instances.size());
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
        }     // ShaderScope automatically unbinds here

        LOG_TRACE("Rendered {} camera frustums", visible_instances.size());
        return {};
    }

    Result<int> CameraFrustumRenderer::pickCamera(const std::vector<std::shared_ptr<const Camera>>& cameras,
                                                  const glm::vec2& mouse_pos,
                                                  const glm::vec2& viewport_pos,
                                                  const glm::vec2& viewport_size,
                                                  const glm::mat4& view,
                                                  const glm::mat4& projection,
                                                  float scale) {
        if (!initialized_ || cameras.empty()) {
            return -1;
        }

        // Use cached instances if available, don't regenerate!
        if (cached_instances_.empty() || camera_ids_.size() != cameras.size()) {
            // Only regenerate if we really have to (first pick or camera count changed)
            LOG_WARN("No cached instances for picking, regenerating");

            // Extract view position for visibility calculation
            glm::vec3 view_position = glm::vec3(glm::inverse(view)[3]);

            // Use the same colors as last render to avoid visual changes
            prepareInstances(cameras, scale, last_train_color_, last_eval_color_, false, view_position);

            if (cached_instances_.empty()) {
                LOG_ERROR("Failed to prepare instances for picking");
                return -1;
            }
        } else {
            LOG_TRACE("Using {} cached instances for picking", cached_instances_.size());
        }

        // Resize picking FBO if needed
        int vp_width = static_cast<int>(viewport_size.x);
        int vp_height = static_cast<int>(viewport_size.y);

        if (vp_width != picking_fbo_width_ || vp_height != picking_fbo_height_) {
            LOG_DEBUG("Resizing picking FBO from {}x{} to {}x{}",
                      picking_fbo_width_, picking_fbo_height_, vp_width, vp_height);
            picking_fbo_width_ = vp_width;
            picking_fbo_height_ = vp_height;

            glBindTexture(GL_TEXTURE_2D, picking_color_texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, picking_fbo_width_, picking_fbo_height_,
                         0, GL_RGB, GL_FLOAT, nullptr);

            glBindTexture(GL_TEXTURE_2D, picking_depth_texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, picking_fbo_width_, picking_fbo_height_,
                         0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        }

        // Save current FBO and viewport
        GLint current_fbo;
        GLint current_viewport[4];
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);
        glGetIntegerv(GL_VIEWPORT, current_viewport);

        // Bind picking FBO
        glBindFramebuffer(GL_FRAMEBUFFER, picking_fbo_);
        glViewport(0, 0, picking_fbo_width_, picking_fbo_height_);

        // Clear to black (ID = 0)
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Render with picking shader
        {
            ShaderScope shader(shader_);

            if (!shader.isBound()) {
                glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
                glViewport(current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]);
                return std::unexpected("Failed to bind picking shader");
            }

            glm::mat4 view_proj = projection * view;
            glm::vec3 view_pos = glm::vec3(glm::inverse(view)[3]);

            shader->set("viewProj", view_proj);
            shader->set("viewPos", view_pos);
            shader->set("pickingMode", true);   // Enable picking mode
            shader->set("enableShading", true); // Render solid faces only

            // Set minimum pick distance based on scale - don't pick frustums too close
            float min_pick_distance = scale * 2.0f; // Adjust this value as needed
            shader->set("minimumPickDistance", min_pick_distance);

            VAOBinder vao_bind(vao_);

            // Upload instance data - USE CACHED INSTANCES
            {
                BufferBinder<GL_ARRAY_BUFFER> instance_bind(instance_vbo_);
                upload_buffer(GL_ARRAY_BUFFER, std::span(cached_instances_), GL_DYNAMIC_DRAW);

                // Setup instance attributes
                for (int i = 0; i < 4; ++i) {
                    glEnableVertexAttribArray(1 + i);
                    glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                          reinterpret_cast<void*>(sizeof(glm::vec4) * i));
                    glVertexAttribDivisor(1 + i, 1);
                }

                // Instance color and alpha (location 5) - now vec4
                glEnableVertexAttribArray(5);
                glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                      reinterpret_cast<void*>(offsetof(InstanceData, color)));
                glVertexAttribDivisor(5, 1);
            }

            // Enable depth testing
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);

            // Draw solid faces only for picking
            {
                BufferBinder<GL_ELEMENT_ARRAY_BUFFER> face_bind(face_ebo_);
                glDrawElementsInstanced(GL_TRIANGLES, num_face_indices_, GL_UNSIGNED_INT, 0, cached_instances_.size());
            }

            // Check for errors
            GLenum err = glGetError();
            if (err != GL_NO_ERROR) {
                LOG_ERROR("OpenGL error during picking render: 0x{:x}", err);
            }

            // Cleanup attributes
            for (int i = 1; i <= 5; ++i) {
                glDisableVertexAttribArray(i);
                if (i >= 1 && i <= 5) {
                    glVertexAttribDivisor(i, 0);
                }
            }
        }

        // Ensure rendering is complete before reading pixels
        glFinish();

        // Read pixel under mouse
        // Convert mouse position relative to viewport
        int pixel_x = static_cast<int>(mouse_pos.x - viewport_pos.x);
        int pixel_y = static_cast<int>(viewport_size.y - (mouse_pos.y - viewport_pos.y)); // Flip Y

        // Clamp to viewport bounds
        pixel_x = std::clamp(pixel_x, 0, picking_fbo_width_ - 1);
        pixel_y = std::clamp(pixel_y, 0, picking_fbo_height_ - 1);

        // Read a small area around the mouse position for debugging
        const int sample_size = 3;
        std::vector<float> pixels(sample_size * sample_size * 3);
        int read_x = std::max(0, pixel_x - 1);
        int read_y = std::max(0, pixel_y - 1);
        int read_width = std::min(sample_size, picking_fbo_width_ - read_x);
        int read_height = std::min(sample_size, picking_fbo_height_ - read_y);

        glReadPixels(read_x, read_y, read_width, read_height, GL_RGB, GL_FLOAT, pixels.data());

        // Get the center pixel (or first pixel if we're at the edge)
        int center_idx = 0;
        if (read_width == 3 && read_height == 3) {
            center_idx = 4 * 3; // Center of 3x3 is index 4
        } else if (read_width >= 2 && read_height >= 2) {
            // Try to get a center-ish pixel
            center_idx = ((read_height / 2) * read_width + (read_width / 2)) * 3;
        }

        float r = pixels[center_idx];
        float g = pixels[center_idx + 1];
        float b = pixels[center_idx + 2];

        // Decode ID from color
        int id = static_cast<int>(r * 255.0f + 0.5f) << 16 |
                 static_cast<int>(g * 255.0f + 0.5f) << 8 |
                 static_cast<int>(b * 255.0f + 0.5f);

        id -= 1; // We added 1 in the shader to avoid 0

        LOG_TRACE("Picked pixel at ({}, {}): RGB({:.3f}, {:.3f}, {:.3f}) -> ID {}",
                  pixel_x, pixel_y, r, g, b, id);

        // Restore previous FBO and viewport
        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
        glViewport(current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]);

        // Return camera ID if valid
        if (id >= 0 && id < static_cast<int>(camera_ids_.size())) {
            LOG_TRACE("Picked camera at index {} with ID {}", id, camera_ids_[id]);
            return camera_ids_[id];
        }

        return -1;
    }

} // namespace gs::rendering