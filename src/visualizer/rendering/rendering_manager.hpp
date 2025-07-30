#pragma once

#include "rendering/renderer.hpp"
#include "rendering/rendering_pipeline.hpp"
#include "rendering/shader.hpp"
#include "scene/scene_manager.hpp"
#include <memory>

namespace gs::visualizer {

    struct RenderSettings {
        float fov = 60.0f;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        bool show_crop_box = false;
        bool use_crop_box = false;
    };

    class RenderingManager {
    public:
        struct RenderContext {
            const Viewport& viewport;
            const RenderSettings& settings;
            const RenderBoundingBox* crop_box;
        };

        RenderingManager();
        ~RenderingManager();

        // Initialize rendering resources
        void initialize();

        // Main render function
        void renderFrame(const RenderContext& context, SceneManager* scene_manager);

        // Settings
        void updateSettings(const RenderSettings& settings) { settings_ = settings; }
        const RenderSettings& getSettings() const { return settings_; }

        // Get shader for external use (crop box rendering)
        std::shared_ptr<Shader> getQuadShader() const { return quad_shader_; }

    private:
        void initializeShaders();
        void drawSceneFrame(const RenderContext& context, SceneManager* scene_manager);
        void drawCropBox(const RenderContext& context);

        RenderSettings settings_;
        std::shared_ptr<ScreenQuadRenderer> screen_renderer_;
        std::shared_ptr<Shader> quad_shader_;
        bool initialized_ = false;
    };

} // namespace gs::visualizer