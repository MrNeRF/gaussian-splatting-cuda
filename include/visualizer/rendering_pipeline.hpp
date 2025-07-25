#pragma once

#include "core/camera.hpp"
#include "core/rasterizer.hpp"
#include "core/splat_data.hpp"
#include "visualizer/renderer.hpp"
#include <glm/glm.hpp>
#include <torch/torch.h>

namespace gs {

    class RenderingPipeline {
    public:
        struct RenderRequest {
            glm::mat3 view_rotation;
            glm::vec3 view_translation;
            glm::ivec2 viewport_size;
            float fov = 60.0f;
            float scaling_modifier = 1.0f;
            bool antialiasing = false;
            RenderMode render_mode = RenderMode::RGB;
        };

        struct RenderResult {
            torch::Tensor image; // [C, H, W] format
            torch::Tensor depth; // Optional depth
            bool valid = false;
        };

        RenderingPipeline();

        // Main render function
        RenderResult render(const SplatData& model, const RenderRequest& request);

        // Upload result to screen renderer
        void uploadToScreen(const RenderResult& result,
                            ScreenQuadRenderer& renderer,
                            const glm::ivec2& viewport_size);

    private:
        Camera createCamera(const RenderRequest& request);
        glm::vec2 computeFov(float fov_degrees, int width, int height);

        torch::Tensor background_;
    };

} // namespace gs
