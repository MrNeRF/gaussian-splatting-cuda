#include "rendering/render_infinite_grid.hpp"
#include "internal/resource_paths.hpp"
#include <iostream>
#include <random>
#include <vector>

namespace gs {

    RenderInfiniteGrid::RenderInfiniteGrid() = default;

    RenderInfiniteGrid::~RenderInfiniteGrid() {
        cleanup();
    }

    void RenderInfiniteGrid::init() {
        if (initialized_)
            return;

        try {
            // Create shader for infinite grid rendering
            shader_ = std::make_unique<Shader>(
                (visualizer::getShaderPath("infinite_grid.vert")).string().c_str(),
                (visualizer::getShaderPath("infinite_grid.frag")).string().c_str(),
                false); // Don't use shader's buffer management

            // Generate OpenGL objects
            glGenVertexArrays(1, &vao_);
            glGenBuffers(1, &vbo_);

            glBindVertexArray(vao_);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_);

            // Full-screen quad vertices (-1 to 1)
            float vertices[] = {
                -1.0f, -1.0f,
                1.0f, -1.0f,
                -1.0f, 1.0f,
                1.0f, 1.0f};

            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

            // Set up vertex attributes
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

            glBindVertexArray(0);

            // Create blue noise texture
            createBlueNoiseTexture();

            initialized_ = true;

        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize InfiniteGrid: " << e.what() << std::endl;
            cleanup();
        }
    }

    void RenderInfiniteGrid::cleanup() {
        if (vao_ != 0) {
            glDeleteVertexArrays(1, &vao_);
            vao_ = 0;
        }
        if (vbo_ != 0) {
            glDeleteBuffers(1, &vbo_);
            vbo_ = 0;
        }
        if (blue_noise_texture_ != 0) {
            glDeleteTextures(1, &blue_noise_texture_);
            blue_noise_texture_ = 0;
        }

        shader_.reset();
        initialized_ = false;
    }

    void RenderInfiniteGrid::createBlueNoiseTexture() {
        const int size = 32;
        std::vector<float> noise_data(size * size);

        // Generate blue noise pattern (simplified version)
        std::mt19937 rng(42); // Fixed seed for consistency
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = 0; i < size * size; ++i) {
            noise_data[i] = dist(rng);
        }

        // Create texture
        glGenTextures(1, &blue_noise_texture_);
        glBindTexture(GL_TEXTURE_2D, blue_noise_texture_);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, size, size, 0, GL_RED, GL_FLOAT, noise_data.data());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void RenderInfiniteGrid::calculateFrustumCorners(const glm::mat4& inv_viewproj,
                                                     glm::vec3& near_origin, glm::vec3& near_x, glm::vec3& near_y,
                                                     glm::vec3& far_origin, glm::vec3& far_x, glm::vec3& far_y) {
        // Transform NDC corners to world space
        auto unproject = [&inv_viewproj](float x, float y, float z) -> glm::vec3 {
            glm::vec4 p = inv_viewproj * glm::vec4(x, y, z, 1.0f);
            return glm::vec3(p) / p.w;
        };

        // Near plane corners in NDC
        glm::vec3 near_bl = unproject(-1.0f, -1.0f, -1.0f);
        glm::vec3 near_br = unproject(1.0f, -1.0f, -1.0f);
        glm::vec3 near_tl = unproject(-1.0f, 1.0f, -1.0f);

        // Far plane corners in NDC
        glm::vec3 far_bl = unproject(-1.0f, -1.0f, 1.0f);
        glm::vec3 far_br = unproject(1.0f, -1.0f, 1.0f);
        glm::vec3 far_tl = unproject(-1.0f, 1.0f, 1.0f);

        // Calculate origins and axes
        near_origin = near_bl;
        near_x = near_br - near_bl;
        near_y = near_tl - near_bl;

        far_origin = far_bl;
        far_x = far_br - far_bl;
        far_y = far_tl - far_bl;
    }

    void RenderInfiniteGrid::render(const glm::mat4& view, const glm::mat4& projection) {
        if (!initialized_ || !shader_)
            return;

        // Calculate matrices
        glm::mat4 viewProj = projection * view;
        glm::mat4 invViewProj = glm::inverse(viewProj);

        // Calculate frustum corners
        glm::vec3 near_origin, near_x, near_y, far_origin, far_x, far_y;
        calculateFrustumCorners(invViewProj, near_origin, near_x, near_y, far_origin, far_x, far_y);

        // Camera position in world space (from view matrix)
        glm::mat4 viewInv = glm::inverse(view);
        glm::vec3 view_position = glm::vec3(viewInv[3]);

        // Save current OpenGL state
        GLboolean depth_mask;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depth_mask);
        GLint blend_src, blend_dst;
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &blend_src);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &blend_dst);
        GLboolean blend_enabled = glIsEnabled(GL_BLEND);

        // Set rendering state
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_TRUE);

        // Bind shader and set uniforms
        shader_->bind();

        shader_->set_uniform("near_origin", near_origin);
        shader_->set_uniform("near_x", near_x);
        shader_->set_uniform("near_y", near_y);
        shader_->set_uniform("far_origin", far_origin);
        shader_->set_uniform("far_x", far_x);
        shader_->set_uniform("far_y", far_y);

        shader_->set_uniform("view_position", view_position);
        shader_->set_uniform("matrix_viewProjection", viewProj);
        shader_->set_uniform("plane", static_cast<int>(plane_));
        shader_->set_uniform("opacity", opacity_);

        // Bind blue noise texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, blue_noise_texture_);
        shader_->set_uniform("blueNoiseTex32", 0);

        // Render the grid
        glBindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        shader_->unbind();

        // Restore OpenGL state
        glDepthMask(depth_mask);
        if (!blend_enabled) {
            glDisable(GL_BLEND);
        } else {
            glBlendFunc(blend_src, blend_dst);
        }
    }

} // namespace gs
