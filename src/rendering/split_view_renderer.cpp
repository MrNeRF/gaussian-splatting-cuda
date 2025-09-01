/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_renderer.hpp"
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "gl_state_guard.hpp"
#include <glad/glad.h>

namespace gs::rendering {

    Result<void> SplitViewRenderer::initialize() {
        if (initialized_) {
            return {};
        }

        LOG_DEBUG("Initializing SplitViewRenderer");

        // Load split view shader
        auto shader_result = load_shader("split_view", "split_view.vert", "split_view.frag", false);
        if (!shader_result) {
            LOG_ERROR("Failed to load split view shader: {}", shader_result.error().what());
            return std::unexpected("Failed to load split view shader");
        }
        split_shader_ = std::move(*shader_result);

        // Setup the quad for rendering
        if (auto result = setupQuad(); !result) {
            return result;
        }

        initialized_ = true;
        LOG_DEBUG("SplitViewRenderer initialized successfully");
        return {};
    }

    Result<void> SplitViewRenderer::setupQuad() {
        // Create VAO and VBO for full-screen quad
        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected("Failed to create VAO");
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            return std::unexpected("Failed to create VBO");
        }
        quad_vbo_ = std::move(*vbo_result);

        // Full-screen quad vertices with texture coordinates
        float quad_vertices[] = {
            // positions   // texCoords
            -1.0f, 1.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f, 0.0f,

            -1.0f, 1.0f, 0.0f, 1.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f};

        // Build VAO
        VAOBuilder builder(std::move(*vao_result));
        std::span<const float> vertices_span(quad_vertices, 24);

        builder.attachVBO(quad_vbo_, vertices_span, GL_STATIC_DRAW)
            .setAttribute({.index = 0, .size = 2, .type = GL_FLOAT, .stride = 4 * sizeof(float), .offset = nullptr})
            .setAttribute({.index = 1, .size = 2, .type = GL_FLOAT, .stride = 4 * sizeof(float), .offset = (void*)(2 * sizeof(float))});

        quad_vao_ = builder.build();
        LOG_DEBUG("Created full-screen quad VAO");
        return {};
    }

    Result<void> SplitViewRenderer::createFramebuffers(int width, int height) {
        // Always check and resize if dimensions don't match
        if (!left_framebuffer_) {
            left_framebuffer_ = std::make_unique<FrameBuffer>();
            LOG_DEBUG("Created left framebuffer");
        }

        if (!right_framebuffer_) {
            right_framebuffer_ = std::make_unique<FrameBuffer>();
            LOG_DEBUG("Created right framebuffer");
        }

        // Always resize if dimensions don't match current size
        if (left_framebuffer_->getWidth() != width || left_framebuffer_->getHeight() != height) {
            LOG_DEBUG("Resizing left framebuffer from {}x{} to {}x{}",
                      left_framebuffer_->getWidth(), left_framebuffer_->getHeight(), width, height);
            left_framebuffer_->resize(width, height);
        }

        if (right_framebuffer_->getWidth() != width || right_framebuffer_->getHeight() != height) {
            LOG_DEBUG("Resizing right framebuffer from {}x{} to {}x{}",
                      right_framebuffer_->getWidth(), right_framebuffer_->getHeight(), width, height);
            right_framebuffer_->resize(width, height);
        }

        return {};
    }

    Result<RenderResult> SplitViewRenderer::render(
        const SplitViewRequest& request,
        RenderingPipeline& pipeline,
        ScreenQuadRenderer& screen_renderer,
        ManagedShader& quad_shader) {

        LOG_TIMER_TRACE("SplitViewRenderer::render");

        if (!initialized_) {
            if (auto result = initialize(); !result) {
                return std::unexpected("Failed to initialize split view renderer");
            }
        }

        if (request.panels.size() != 2) {
            return std::unexpected("Split view requires exactly 2 panels");
        }

        const auto* left_model = request.panels[0].model;
        const auto* right_model = request.panels[1].model;

        if (!left_model || !right_model) {
            return std::unexpected("Invalid models for split view");
        }

        LOG_DEBUG("Split view: left='{}' ({}), right='{}' ({}), split={}",
                  request.panels[0].label, left_model->size(),
                  request.panels[1].label, right_model->size(),
                  request.panels[0].end_position);

        // Create/resize framebuffers if needed
        if (auto result = createFramebuffers(request.viewport.size.x, request.viewport.size.y);
            !result) {
            return std::unexpected(result.error());
        }

        // Save current OpenGL state
        GLint current_viewport[4];
        glGetIntegerv(GL_VIEWPORT, current_viewport);
        GLint current_fbo;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);

        LOG_TRACE("Current viewport: [{}, {}, {}, {}], FBO: {}",
                  current_viewport[0], current_viewport[1],
                  current_viewport[2], current_viewport[3], current_fbo);

        // Create render request for both panels
        RenderingPipeline::RenderRequest base_req{
            .view_rotation = request.viewport.rotation,
            .view_translation = request.viewport.translation,
            .viewport_size = request.viewport.size,
            .fov = request.viewport.fov,
            .scaling_modifier = request.scaling_modifier,
            .antialiasing = request.antialiasing,
            .render_mode = RenderMode::RGB,
            .crop_box = nullptr,
            .background_color = request.background_color,
            .point_cloud_mode = request.point_cloud_mode,
            .voxel_size = request.voxel_size,
            .gut = request.gut};

        // Handle crop box if present
        std::unique_ptr<geometry::BoundingBox> temp_crop_box;
        if (request.crop_box.has_value()) {
            temp_crop_box = std::make_unique<geometry::BoundingBox>();
            temp_crop_box->setBounds(request.crop_box->min, request.crop_box->max);
            geometry::EuclideanTransform transform(request.crop_box->transform);
            temp_crop_box->setworld2BBox(transform);
            base_req.crop_box = temp_crop_box.get();
        }

        // === STEP 1: Render LEFT model to framebuffer ===
        LOG_DEBUG("Rendering left panel '{}' to framebuffer", request.panels[0].label);

        left_framebuffer_->bind();
        glViewport(0, 0, request.viewport.size.x, request.viewport.size.y);
        glClearColor(request.background_color.r, request.background_color.g, request.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        auto left_result = pipeline.render(*left_model, base_req);
        if (!left_result) {
            LOG_ERROR("Failed to render left model: {}", left_result.error());
            left_framebuffer_->unbind();
            return std::unexpected(left_result.error());
        }

        // Present to left framebuffer
        if (auto upload_result = RenderingPipeline::uploadToScreen(*left_result, screen_renderer, request.viewport.size);
            !upload_result) {
            LOG_ERROR("Failed to upload left model: {}", upload_result.error());
        } else {
            glViewport(0, 0, request.viewport.size.x, request.viewport.size.y);
            if (auto render_result = screen_renderer.render(quad_shader); !render_result) {
                LOG_ERROR("Failed to render left to framebuffer: {}", render_result.error());
            }
        }

        left_framebuffer_->unbind();

        // === STEP 2: Render RIGHT model to framebuffer ===
        LOG_DEBUG("Rendering right panel '{}' to framebuffer", request.panels[1].label);

        right_framebuffer_->bind();
        glViewport(0, 0, request.viewport.size.x, request.viewport.size.y);
        glClearColor(request.background_color.r, request.background_color.g, request.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        auto right_result = pipeline.render(*right_model, base_req);
        if (!right_result) {
            LOG_ERROR("Failed to render right model: {}", right_result.error());
            right_framebuffer_->unbind();
            return std::unexpected(right_result.error());
        }

        // Present to right framebuffer
        if (auto upload_result = RenderingPipeline::uploadToScreen(*right_result, screen_renderer, request.viewport.size);
            !upload_result) {
            LOG_ERROR("Failed to upload right model: {}", upload_result.error());
        } else {
            glViewport(0, 0, request.viewport.size.x, request.viewport.size.y);
            if (auto render_result = screen_renderer.render(quad_shader); !render_result) {
                LOG_ERROR("Failed to render right to framebuffer: {}", render_result.error());
            }
        }

        right_framebuffer_->unbind();

        // === STEP 3: Composite the two views directly to screen ===
        LOG_DEBUG("Compositing split view at position {}", request.panels[0].end_position);

        // Render directly to screen (framebuffer 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Set viewport to the actual screen area we want to render to
        glViewport(current_viewport[0], current_viewport[1],
                   current_viewport[2], current_viewport[3]);

        // Clear the screen before compositing
        glClearColor(request.background_color.r, request.background_color.g,
                     request.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use split view shader to composite directly to screen
        if (auto result = compositeSplitView(
                left_framebuffer_->getFrameTexture(),
                right_framebuffer_->getFrameTexture(),
                request.panels[0].end_position,
                request.divider_color);
            !result) {
            LOG_ERROR("Failed to composite split view: {}", result.error());
            // Restore framebuffer binding before returning
            glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
            return std::unexpected(result.error());
        }

        // Restore the original framebuffer binding
        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);

        // Return a result (using left as representative)
        return RenderResult{
            .image = std::make_shared<torch::Tensor>(left_result->image),
            .depth = std::make_shared<torch::Tensor>(left_result->depth)};
    }

    Result<void> SplitViewRenderer::compositeSplitView(
        GLuint left_texture,
        GLuint right_texture,
        float split_position,
        const glm::vec4& divider_color) {

        LOG_TRACE("Compositing: left_tex={}, right_tex={}, split={}",
                  left_texture, right_texture, split_position);

        // Save current state
        GLint current_viewport[4];
        glGetIntegerv(GL_VIEWPORT, current_viewport);

        // Disable depth test for compositing
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Use the split view shader
        if (auto result = split_shader_.bind(); !result) {
            LOG_ERROR("Failed to bind split shader: {}", result.error());
            return result;
        }

        // Set textures
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, left_texture);
        if (auto result = split_shader_.set("leftTexture", 0); !result) {
            LOG_ERROR("Failed to set leftTexture uniform: {}", result.error());
        }

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, right_texture);
        if (auto result = split_shader_.set("rightTexture", 1); !result) {
            LOG_ERROR("Failed to set rightTexture uniform: {}", result.error());
        }

        // Set other uniforms
        if (auto result = split_shader_.set("splitPosition", split_position); !result) {
            LOG_ERROR("Failed to set splitPosition uniform: {}", result.error());
        }

        // Use float for bool uniform (GLSL converts non-zero to true)
        if (auto result = split_shader_.set("showDivider", 1.0f); !result) {
            LOG_ERROR("Failed to set showDivider uniform: {}", result.error());
        }

        if (auto result = split_shader_.set("dividerColor", divider_color); !result) {
            LOG_ERROR("Failed to set dividerColor uniform: {}", result.error());
        }

        // Draw the full-screen quad
        VAOBinder vao_bind(quad_vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        LOG_TRACE("Drew composite quad");

        split_shader_.unbind();

        // Restore state
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);

        return {};
    }

} // namespace gs::rendering