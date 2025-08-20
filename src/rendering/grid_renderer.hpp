#pragma once

#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>

namespace gs::rendering {

    class RenderInfiniteGrid {
    public:
        enum class GridPlane {
            YZ = 0, // X plane (YZ grid)
            XZ = 1, // Y plane (XZ grid)
            XY = 2  // Z plane (XY grid)
        };

        RenderInfiniteGrid() = default;
        ~RenderInfiniteGrid() = default;

        // Initialize OpenGL resources - now returns Result
        Result<void> init();

        // Check if initialized
        bool isInitialized() const { return initialized_; }

        // Render the infinite grid - now returns Result
        Result<void> render(const glm::mat4& view, const glm::mat4& projection);

        // Set grid parameters
        void setOpacity(float opacity) { opacity_ = glm::clamp(opacity, 0.0f, 1.0f); }
        void setFadeDistance(float near_dist, float far_dist) {
            fade_start_ = near_dist;
            fade_end_ = far_dist;
        }
        void setPlane(GridPlane plane) { plane_ = plane; }
        GridPlane getPlane() const { return plane_; }

    private:
        Result<void> createBlueNoiseTexture();
        void calculateFrustumCorners(const glm::mat4& inv_viewproj,
                                     glm::vec3& near_origin, glm::vec3& near_x, glm::vec3& near_y,
                                     glm::vec3& far_origin, glm::vec3& far_x, glm::vec3& far_y);

        // OpenGL resources using RAII
        ManagedShader shader_;
        VAO vao_;
        VBO vbo_;
        Texture blue_noise_texture_;

        // Grid parameters
        GridPlane plane_ = GridPlane::XZ;
        float opacity_ = 1.0f;
        float fade_start_ = 1000.0f;
        float fade_end_ = 5000.0f;

        bool initialized_ = false;
    };

} // namespace gs::rendering
