#include "gui/viewport_gizmo.hpp"
#include "gui/text_renderer.hpp"
#include "internal/resource_paths.hpp"
#include "rendering/shader.hpp"
#include <algorithm>
#include <iostream>
#include <numbers>
#include <ranges>
#include <vector>

namespace gs::gui {

    constexpr glm::vec3 ViewportGizmo::colors_[];

    ViewportGizmo::ViewportGizmo() = default;

    ViewportGizmo::~ViewportGizmo() {
        shutdown();
    }

    void ViewportGizmo::initialize() {
        if (initialized_)
            return;

        createShaders();
        generateGeometry();

        // Initialize text renderer using actual window size (will be updated in render)
        int width = 1280, height = 720;
        GLFWwindow* window = glfwGetCurrentContext();
        if (window) {
            glfwGetFramebufferSize(window, &width, &height);
        }
        text_renderer_ = std::make_unique<TextRenderer>(width, height);

        // Load font from our assets
        std::string font_path = std::string(PROJECT_ROOT_PATH) +
                                "/src/visualizer/resources/assets/JetBrainsMono-Regular.ttf";
        if (!text_renderer_->LoadFont(font_path, 48)) {
            std::cerr << "ViewportGizmo: Failed to load font!" << std::endl;
            text_renderer_.reset();
        }

        initialized_ = true;
    }

    void ViewportGizmo::shutdown() {
        if (vao_) {
            glDeleteVertexArrays(1, &vao_);
            vao_ = 0;
        }
        if (vbo_) {
            glDeleteBuffers(1, &vbo_);
            vbo_ = 0;
        }
        shader_.reset();
        text_renderer_.reset();
        initialized_ = false;
    }

    void ViewportGizmo::createShaders() {
        // Use the shader system to create shaders
        try {
            shader_ = std::make_unique<gs::rendering::Shader>(
                (gs::visualizer::getShaderPath("viewport_gizmo.vert")).string().c_str(),
                (gs::visualizer::getShaderPath("viewport_gizmo.frag")).string().c_str(),
                false // Don't create buffer
            );
            std::cout << "[ViewportGizmo] Shaders created successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[ViewportGizmo] ERROR creating shaders: " << e.what() << std::endl;
            throw;
        }
    }

    void ViewportGizmo::generateGeometry() {
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);

        std::vector<float> vertices;
        vertices.reserve(10000);

        auto addVertex = [&](float x, float y, float z, float nx, float ny, float nz) {
            vertices.insert(vertices.end(), {x, y, z, nx, ny, nz});
        };

        // Generate cylinder (for axes)
        constexpr int segments = 16;
        constexpr size_t kVertexBufferReserve = segments * 6; // Number of vertices for cylinder
        vertices.reserve(kVertexBufferReserve);

        // Generate cylinder (for axes)
        constexpr float two_pi = 2 * std::numbers::pi_v<float>;

        for (const auto i : std::views::iota(0, segments)) {
            float a1 = static_cast<float>(i) / segments * two_pi;
            float a2 = static_cast<float>(i + 1) / segments * two_pi;
            float c1 = cos(a1), s1 = sin(a1);
            float c2 = cos(a2), s2 = sin(a2);

            addVertex(c1, s1, 0, c1, s1, 0);
            addVertex(c2, s2, 0, c2, s2, 0);
            addVertex(c1, s1, 1, c1, s1, 0);

            addVertex(c2, s2, 0, c2, s2, 0);
            addVertex(c2, s2, 1, c2, s2, 0);
            addVertex(c1, s1, 1, c1, s1, 0);
        }
        cylinder_vertex_count_ = segments * 6;

        // Generate sphere
        sphere_vertex_start_ = vertices.size() / 6;
        constexpr int latBands = 16, longBands = 16;

        for (const auto lat : std::views::iota(0, latBands)) {
            float theta1 = static_cast<float>(lat) * std::numbers::pi_v<float> / latBands;
            float theta2 = static_cast<float>(lat + 1) * std::numbers::pi_v<float> / latBands;

            float sinTheta1 = sin(theta1), cosTheta1 = cos(theta1);
            float sinTheta2 = sin(theta2), cosTheta2 = cos(theta2);

            for (const auto lon : std::views::iota(0, longBands)) {
                float phi1 = static_cast<float>(lon) * two_pi / longBands;
                float phi2 = static_cast<float>(lon + 1) * two_pi / longBands;

                float sinPhi1 = sin(phi1), cosPhi1 = cos(phi1);
                float sinPhi2 = sin(phi2), cosPhi2 = cos(phi2);

                float x1 = sinTheta1 * cosPhi1, y1 = cosTheta1, z1 = sinTheta1 * sinPhi1;
                float x2 = sinTheta1 * cosPhi2, y2 = cosTheta1, z2 = sinTheta1 * sinPhi2;
                float x3 = sinTheta2 * cosPhi2, y3 = cosTheta2, z3 = sinTheta2 * sinPhi2;
                float x4 = sinTheta2 * cosPhi1, y4 = cosTheta2, z4 = sinTheta2 * sinPhi1;

                addVertex(x1, y1, z1, x1, y1, z1);
                addVertex(x2, y2, z2, x2, y2, z2);
                addVertex(x3, y3, z3, x3, y3, z3);

                addVertex(x1, y1, z1, x1, y1, z1);
                addVertex(x3, y3, z3, x3, y3, z3);
                addVertex(x4, y4, z4, x4, y4, z4);
            }
        }
        sphere_vertex_count_ = (vertices.size() / 6) - sphere_vertex_start_;

        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glBindVertexArray(0);
    }

    void ViewportGizmo::render(const glm::mat3& camera_rotation,
                               const glm::vec2& viewport_pos,
                               const glm::vec2& viewport_size) {
        if (!initialized_)
            return;

        // Save comprehensive OpenGL state
        GLint vp[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);
        GLboolean blend_enabled = glIsEnabled(GL_BLEND);
        GLboolean stencil_test_enabled = glIsEnabled(GL_STENCIL_TEST);
        GLboolean scissor_test_enabled = glIsEnabled(GL_SCISSOR_TEST);
        GLboolean cull_face_enabled = glIsEnabled(GL_CULL_FACE);
        GLint depth_func;
        glGetIntegerv(GL_DEPTH_FUNC, &depth_func);
        GLint blend_src, blend_dst;
        glGetIntegerv(GL_BLEND_SRC, &blend_src);
        glGetIntegerv(GL_BLEND_DST, &blend_dst);
        GLint blend_equation_rgb, blend_equation_alpha;
        glGetIntegerv(GL_BLEND_EQUATION_RGB, &blend_equation_rgb);
        glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &blend_equation_alpha);
        GLboolean depth_mask;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depth_mask);
        GLboolean color_mask[4];
        glGetBooleanv(GL_COLOR_WRITEMASK, color_mask);
        GLint active_texture;
        glGetIntegerv(GL_ACTIVE_TEXTURE, &active_texture);

        // Calculate gizmo position (upper right of viewport)
        int gizmo_x = static_cast<int>(viewport_pos.x + viewport_size.x - size_ - margin_);
        int gizmo_y = static_cast<int>(viewport_pos.y + margin_);

        // Set gizmo viewport
        glViewport(gizmo_x, gizmo_y, size_, size_);

        // Clear depth for gizmo
        glEnable(GL_SCISSOR_TEST);
        glScissor(gizmo_x, gizmo_y, size_, size_);
        glClear(GL_DEPTH_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);

        // Enable depth testing for gizmo
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);

        // Disable face culling for gizmo
        glDisable(GL_CULL_FACE);

        // Fixed gizmo camera
        glm::mat4 baseView = glm::lookAt(glm::vec3(1.8f, 1.35f, 1.8f), glm::vec3(0), glm::vec3(0, 1, 0));

        // Apply main camera rotation
        glm::mat4 view = baseView * glm::mat4(glm::transpose(camera_rotation));
        glm::mat4 proj = glm::perspective(glm::radians(35.0f), 1.0f, 0.1f, 10.0f);

        // Calculate reference distance
        glm::vec3 originCamSpace = glm::vec3(view * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        float refDist = glm::length(originCamSpace);

        // Use shader
        shader_->bind();

        // Draw axes
        const float sphereRadius = 0.198f;
        const float axisLen = 0.63f - sphereRadius;
        const float axisRad = 0.0225f;
        const float labelDistance = 0.63f;

        const glm::mat4 rotations[3] = {
            glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(0, 1, 0)),  // X
            glm::rotate(glm::mat4(1), glm::radians(-90.0f), glm::vec3(1, 0, 0)), // Y
            glm::mat4(1)                                                         // Z
        };

        glBindVertexArray(vao_);

        // Draw axis cylinders
        for (int i = 0; i < 3; i++) {
            glm::mat4 model = rotations[i] * glm::scale(glm::mat4(1), glm::vec3(axisRad, axisRad, axisLen));
            glm::mat4 mvp = proj * view * model;
            shader_->set_uniform("uMVP", mvp);
            shader_->set_uniform("uModel", model);
            shader_->set_uniform("uColor", colors_[i]);
            shader_->set_uniform("uAlpha", 1.0f);
            shader_->set_uniform("uUseLighting", 1);
            glDrawArrays(GL_TRIANGLES, 0, cylinder_vertex_count_);
        }

        // Enable blending for spheres
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBlendEquation(GL_FUNC_ADD);

        // Sphere info for text rendering
        struct SphereInfo {
            glm::vec3 screenPos;
            float depth;
            int index;
            bool visible;
        };
        SphereInfo sphereInfo[3];

        // Draw spheres and calculate positions
        for (int i = 0; i < 3; i++) {
            glm::vec3 labelPos = glm::vec3(0);
            labelPos[i] = labelDistance;

            glm::vec3 camSpacePos = glm::vec3(view * glm::vec4(labelPos, 1.0f));
            float dist = glm::length(camSpacePos);
            float scaleFactor = dist / refDist;

            glm::mat4 model = glm::translate(glm::mat4(1), labelPos) *
                              glm::scale(glm::mat4(1), glm::vec3(sphereRadius * scaleFactor));
            glm::mat4 mvp = proj * view * model;
            shader_->set_uniform("uMVP", mvp);
            shader_->set_uniform("uModel", glm::mat4(1.0f));
            shader_->set_uniform("uColor", colors_[i]);
            shader_->set_uniform("uAlpha", 1.0f);
            shader_->set_uniform("uUseLighting", 0);
            glDrawArrays(GL_TRIANGLES, sphere_vertex_start_, sphere_vertex_count_);

            // Calculate screen position
            glm::vec4 clipPos = proj * view * glm::vec4(labelPos, 1.0f);
            if (clipPos.w > 0) {
                glm::vec3 ndcPos = glm::vec3(clipPos) / clipPos.w;
                float gizmoX = (ndcPos.x * 0.5f + 0.5f) * size_;
                float gizmoY = (ndcPos.y * 0.5f + 0.5f) * size_;

                sphereInfo[i].screenPos.x = gizmoX + gizmo_x;
                sphereInfo[i].screenPos.y = gizmo_y + gizmoY;
                sphereInfo[i].depth = clipPos.z / clipPos.w;
                sphereInfo[i].index = i;
                sphereInfo[i].visible = true;
            } else {
                sphereInfo[i].visible = false;
            }
        }

        // Sort spheres by depth
        std::sort(sphereInfo, sphereInfo + 3, [](const SphereInfo& a, const SphereInfo& b) {
            return a.depth > b.depth;
        });

        // Make screen positions relative to the current viewport origin
        for (int i = 0; i < 3; ++i) {
            sphereInfo[i].screenPos.x -= vp[0];
            sphereInfo[i].screenPos.y -= vp[1];
        }

        // Draw text labels with occlusion
        if (text_renderer_) {
            const char* axisLabels[3] = {"X", "Y", "Z"};

            // Update text renderer size if needed
            int window_width = vp[2];
            int window_height = vp[3];
            text_renderer_->updateScreenSize(window_width, window_height);

            // Ensure proper color mask for text rendering
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

            glEnable(GL_STENCIL_TEST);

            for (int i = 0; i < 3; i++) {
                if (sphereInfo[i].visible) {
                    int idx = sphereInfo[i].index;

                    glClearStencil(0);
                    glClear(GL_STENCIL_BUFFER_BIT);

                    // Mark occluding spheres in stencil
                    glStencilFunc(GL_ALWAYS, 1, 0xFF);
                    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
                    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
                    glDepthMask(GL_FALSE);
                    glDepthFunc(GL_LEQUAL);

                    // Redraw closer spheres
                    for (int j = 0; j < 3; j++) {
                        if (sphereInfo[j].visible && sphereInfo[j].depth < sphereInfo[i].depth) {
                            int jdx = sphereInfo[j].index;
                            glm::vec3 jLabelPos = glm::vec3(0);
                            jLabelPos[jdx] = labelDistance;

                            glm::vec3 jCamSpacePos = glm::vec3(view * glm::vec4(jLabelPos, 1.0f));
                            float jDist = glm::length(jCamSpacePos);
                            float jScaleFactor = jDist / refDist;

                            glm::mat4 jModel = glm::translate(glm::mat4(1), jLabelPos) *
                                               glm::scale(glm::mat4(1), glm::vec3(sphereRadius * jScaleFactor));
                            glm::mat4 jMvp = proj * view * jModel;
                            shader_->set_uniform("uMVP", jMvp);
                            shader_->set_uniform("uModel", glm::mat4(1.0f));
                            shader_->set_uniform("uColor", colors_[jdx]);
                            shader_->set_uniform("uAlpha", 1.0f);
                            shader_->set_uniform("uUseLighting", 0);
                            glDrawArrays(GL_TRIANGLES, sphere_vertex_start_, sphere_vertex_count_);
                        }
                    }

                    glDepthFunc(GL_LESS);
                    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                    glDepthMask(GL_TRUE);
                    glStencilFunc(GL_EQUAL, 0, 0xFF);
                    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

                    // Render text
                    glViewport(vp[0], vp[1], vp[2], vp[3]);

                    float scale = 0.28f;
                    float textWidth = 28.8f * scale;
                    float textHeight = 48.0f * scale;
                    float baselineOffset = textHeight * 0.75f;

                    glDisable(GL_DEPTH_TEST);
                    glDepthMask(GL_FALSE);

                    // Ensure proper texture unit
                    glActiveTexture(GL_TEXTURE0);

                    text_renderer_->RenderText(
                        axisLabels[idx],
                        sphereInfo[i].screenPos.x - textWidth * 0.5f,
                        sphereInfo[i].screenPos.y - baselineOffset + textHeight * 0.5f,
                        scale,
                        glm::vec3(1.0f, 1.0f, 1.0f));

                    glEnable(GL_DEPTH_TEST);
                    glDepthMask(GL_TRUE);
                    glViewport(gizmo_x, gizmo_y, size_, size_);
                }
            }

            glDisable(GL_STENCIL_TEST);
        }

        shader_->unbind();
        glBindVertexArray(0);

        // Restore ALL OpenGL state
        glViewport(vp[0], vp[1], vp[2], vp[3]);

        if (!depth_test_enabled)
            glDisable(GL_DEPTH_TEST);
        else
            glEnable(GL_DEPTH_TEST);

        if (!blend_enabled)
            glDisable(GL_BLEND);
        else
            glEnable(GL_BLEND);

        if (stencil_test_enabled)
            glEnable(GL_STENCIL_TEST);
        else
            glDisable(GL_STENCIL_TEST);

        if (scissor_test_enabled)
            glEnable(GL_SCISSOR_TEST);
        else
            glDisable(GL_SCISSOR_TEST);

        if (cull_face_enabled)
            glEnable(GL_CULL_FACE);
        else
            glDisable(GL_CULL_FACE);

        glDepthFunc(depth_func);
        glBlendFunc(blend_src, blend_dst);
        glBlendEquationSeparate(blend_equation_rgb, blend_equation_alpha);
        glDepthMask(depth_mask);
        glColorMask(color_mask[0], color_mask[1], color_mask[2], color_mask[3]);
        glActiveTexture(active_texture);
    }

} // namespace gs::gui