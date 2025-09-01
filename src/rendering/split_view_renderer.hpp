/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "framebuffer.hpp"
#include "gl_resources.hpp"
#include "rendering/rendering.hpp"
#include "rendering_pipeline.hpp"
#include "shader_manager.hpp"
#include <memory>

namespace gs::rendering {

    class SplitViewRenderer {
    public:
        SplitViewRenderer() = default;
        ~SplitViewRenderer() = default;

        Result<void> initialize();

        Result<RenderResult> render(
            const SplitViewRequest& request,
            RenderingPipeline& pipeline,
            ScreenQuadRenderer& screen_renderer,
            ManagedShader& quad_shader);

    private:
        // Framebuffers for each panel
        std::unique_ptr<FrameBuffer> left_framebuffer_;
        std::unique_ptr<FrameBuffer> right_framebuffer_;

        // Split view compositing shader
        ManagedShader split_shader_;

        // Quad VAO for rendering
        VAO quad_vao_;
        VBO quad_vbo_;

        bool initialized_ = false;

        Result<void> createFramebuffers(int width, int height);
        Result<void> setupQuad();
        Result<void> compositeSplitView(
            GLuint left_texture,
            GLuint right_texture,
            float split_position,
            const glm::vec4& divider_color);
    };

} // namespace gs::rendering