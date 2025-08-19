#include "viewport_gizmo.hpp"
#include "gl_state_guard.hpp"
#include "shader_paths.hpp"
#include "text_renderer.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <format>
#include <iostream>
#include <numbers>
#include <ranges>
#include <vector>

namespace gs::rendering {

    constexpr glm::vec3 ViewportGizmo::colors_[];

    ViewportGizmo::ViewportGizmo() = default;  // Define constructor here
    ViewportGizmo::~ViewportGizmo() = default; // Define destructor here

    Result<void> ViewportGizmo::initialize() {
        if (initialized_)
            return {};

        if (auto result = createShaders(); !result) {
            return result;
        }

        if (auto result = generateGeometry(); !result) {
            return result;
        }

        // Initialize text renderer using actual window size (will be updated in render)
        int width = 1280, height = 720;
        GLFWwindow* window = glfwGetCurrentContext();
        if (window) {
            glfwGetFramebufferSize(window, &width, &height);
        }
        text_renderer_ = std::make_unique<TextRenderer>(width, height);

        // Load font from our assets
        std::string font_path = std::string(PROJECT_ROOT_PATH) +
                                "/src/rendering/resources/assets/JetBrainsMono-Regular.ttf";
        if (auto result = text_renderer_->LoadFont(font_path, 48); !result) {
            std::cerr << "ViewportGizmo: Failed to load font: " << result.error() << std::endl;
            text_renderer_.reset();
        }

        initialized_ = true;
        return {};
    }

    void ViewportGizmo::shutdown() {
        vao_ = VAO();
        vbo_ = VBO();
        shader_ = ManagedShader();
        text_renderer_.reset();
        initialized_ = false;
    }

    Result<void> ViewportGizmo::createShaders() {
        auto result = load_shader("viewport_gizmo", "viewport_gizmo.vert", "viewport_gizmo.frag", false);
        if (!result) {
            return std::unexpected(result.error().what());
        }
        shader_ = std::move(*result);
        std::cout << "[ViewportGizmo] Shaders created successfully" << std::endl;
        return {};
    }

    Result<void> ViewportGizmo::generateGeometry() {
        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected(vao_result.error());
        }
        vao_ = std::move(*vao_result);

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            return std::unexpected(vbo_result.error());
        }
        vbo_ = std::move(*vbo_result);

        VAOBinder vao_bind(vao_);
        BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);

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

        upload_buffer(GL_ARRAY_BUFFER, std::span(vertices), GL_STATIC_DRAW);

        // Position attribute
        VertexAttribute position_attr{
            .index = 0,
            .size = 3,
            .type = GL_FLOAT,
            .normalized = GL_FALSE,
            .stride = 6 * sizeof(float),
            .offset = nullptr};
        position_attr.apply();

        // Normal attribute
        VertexAttribute normal_attr{
            .index = 1,
            .size = 3,
            .type = GL_FLOAT,
            .normalized = GL_FALSE,
            .stride = 6 * sizeof(float),
            .offset = (void*)(3 * sizeof(float))};
        normal_attr.apply();

        return {};
    }

    Result<void> ViewportGizmo::render(const glm::mat3& camera_rotation,
                                       const glm::vec2& viewport_pos,
                                       const glm::vec2& viewport_size) {
        if (!initialized_)
            return std::unexpected("Viewport gizmo not initialized");

        // Use RAII for OpenGL state management
        GLStateGuard state_guard;

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
        ShaderScope s(shader_);

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

        VAOBinder vao_bind(vao_);

        // Draw axis cylinders
        for (int i = 0; i < 3; i++) {
            glm::mat4 model = rotations[i] * glm::scale(glm::mat4(1), glm::vec3(axisRad, axisRad, axisLen));
            glm::mat4 mvp = proj * view * model;
            if (auto result = s->set("uMVP", mvp); !result)
                return result;
            if (auto result = s->set("uModel", model); !result)
                return result;
            if (auto result = s->set("uColor", colors_[i]); !result)
                return result;
            if (auto result = s->set("uAlpha", 1.0f); !result)
                return result;
            if (auto result = s->set("uUseLighting", 1); !result)
                return result;
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
            if (auto result = s->set("uMVP", mvp); !result)
                return result;
            if (auto result = s->set("uModel", glm::mat4(1.0f)); !result)
                return result;
            if (auto result = s->set("uColor", colors_[i]); !result)
                return result;
            if (auto result = s->set("uAlpha", 1.0f); !result)
                return result;
            if (auto result = s->set("uUseLighting", 0); !result)
                return result;
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

        // Get current viewport for relative positioning
        auto vp = state_guard.savedState().viewport;

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
                            s->set("uMVP", jMvp);
                            s->set("uModel", glm::mat4(1.0f));
                            s->set("uColor", colors_[jdx]);
                            s->set("uAlpha", 1.0f);
                            s->set("uUseLighting", 0);
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

                    if (auto result = text_renderer_->RenderText(
                            axisLabels[idx],
                            sphereInfo[i].screenPos.x - textWidth * 0.5f,
                            sphereInfo[i].screenPos.y - baselineOffset + textHeight * 0.5f,
                            scale,
                            glm::vec3(1.0f, 1.0f, 1.0f));
                        !result) {
                        // Continue rendering even if text fails
                        std::cerr << "Failed to render text: " << result.error() << std::endl;
                    }

                    glEnable(GL_DEPTH_TEST);
                    glDepthMask(GL_TRUE);
                    glViewport(gizmo_x, gizmo_y, size_, size_);
                }
            }

            glDisable(GL_STENCIL_TEST);
        }

        // State automatically restored by GLStateGuard destructor
        return {};
    }

} // namespace gs::rendering
