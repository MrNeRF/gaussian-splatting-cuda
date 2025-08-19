#include "grid_renderer.hpp"
#include "shader_paths.hpp"
#include <iostream>
#include <random>
#include <vector>

namespace gs::rendering {

    void RenderInfiniteGrid::init() {
        if (initialized_)
            return;

        try {
            // Create shader for infinite grid rendering
            auto result = load_shader("infinite_grid", "infinite_grid.vert", "infinite_grid.frag", false);
            if (!result) {
                throw std::runtime_error(result.error().what());
            }
            shader_ = std::move(*result);

            // Create OpenGL objects using RAII
            auto vao_result = create_vao();
            if (!vao_result) {
                throw std::runtime_error(vao_result.error().what());
            }
            vao_ = std::move(*vao_result);

            auto vbo_result = create_vbo();
            if (!vbo_result) {
                throw std::runtime_error(vbo_result.error().what());
            }
            vbo_ = std::move(*vbo_result);

            VAOBinder vao_bind(vao_);
            BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);

            // Full-screen quad vertices (-1 to 1)
            float vertices[] = {
                -1.0f, -1.0f,
                1.0f, -1.0f,
                -1.0f, 1.0f,
                1.0f, 1.0f};

            upload_buffer(GL_ARRAY_BUFFER, vertices, 8, GL_STATIC_DRAW);

            // Set up vertex attributes
            VertexAttribute position_attr{
                .index = 0,
                .size = 2,
                .type = GL_FLOAT,
                .normalized = GL_FALSE,
                .stride = 2 * sizeof(float),
                .offset = nullptr};
            position_attr.apply();

            // Create blue noise texture
            createBlueNoiseTexture();

            initialized_ = true;

        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize InfiniteGrid: " << e.what() << std::endl;
            initialized_ = false;
            throw;
        }
    }

    void RenderInfiniteGrid::createBlueNoiseTexture() {
        const int size = 32;
        std::vector<float> noise_data(size * size);

        // Generate white noise pattern using uniform random distribution
        std::mt19937 rng(42); // Fixed seed for consistency
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = 0; i < size * size; ++i) {
            noise_data[i] = dist(rng);
        }

        // Create texture using RAII
        GLuint tex_id;
        glGenTextures(1, &tex_id);
        blue_noise_texture_ = Texture(tex_id);

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
        if (!initialized_ || !shader_.valid())
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
        glGetIntegerv(GL_BLEND_SRC_RGB, &blend_src);
        glGetIntegerv(GL_BLEND_DST_RGB, &blend_dst);
        GLboolean blend_enabled = glIsEnabled(GL_BLEND);

        // Set rendering state
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_TRUE);

        // Bind shader and set uniforms
        ShaderScope s(shader_);

        s->set("near_origin", near_origin);
        s->set("near_x", near_x);
        s->set("near_y", near_y);
        s->set("far_origin", far_origin);
        s->set("far_x", far_x);
        s->set("far_y", far_y);

        s->set("view_position", view_position);
        s->set("matrix_viewProjection", viewProj);
        s->set("plane", static_cast<int>(plane_));
        s->set("opacity", opacity_);

        // Bind blue noise texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, blue_noise_texture_);
        s->set("blueNoiseTex32", 0);

        // Render the grid
        VAOBinder vao_bind(vao_);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // Restore OpenGL state
        glDepthMask(depth_mask);
        if (!blend_enabled) {
            glDisable(GL_BLEND);
        } else {
            glBlendFunc(blend_src, blend_dst);
        }
    }

} // namespace gs::rendering