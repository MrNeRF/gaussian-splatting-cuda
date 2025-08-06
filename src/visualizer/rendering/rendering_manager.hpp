#pragma once

#include "rendering/renderer.hpp"
#include "rendering/rendering_pipeline.hpp"
#include "rendering/shader.hpp"
#include "scene/scene_manager.hpp"
#include <memory>

namespace gs {
    class RenderCoordinateAxes;
}

namespace gs::visualizer {

    struct RenderSettings {
        float fov = 60.0f;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        bool show_crop_box = false;
        bool use_crop_box = false;
        bool show_coord_axes = false;
    };

    struct ViewportRegion {
        float x, y, width, height;
    };

    class RenderingManager {
    public:
        struct RenderContext {
            const Viewport& viewport;
            const RenderSettings& settings;
            const RenderBoundingBox* crop_box;
            const RenderCoordinateAxes* coord_axes;
            const geometry::EuclideanTransform* world_to_user;
            const ViewportRegion* viewport_region = nullptr;
            bool has_focus = false; // Indicates if the viewport has focus for input handling
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
        void drawSceneFrame(const RenderContext& context, SceneManager* scene_manager, const Viewport& render_viewport);
        void drawFocusIndicator(const RenderContext& context);
        void drawCropBox(const RenderContext& context, const Viewport& render_viewport);
        void drawSceneFrame(const RenderContext& context, SceneManager* scene_manager);
        void drawCropBox(const RenderContext& context);
        void drawCoordAxes(const RenderContext& context);


        RenderSettings settings_;
        std::shared_ptr<ScreenQuadRenderer> screen_renderer_;
        std::shared_ptr<Shader> quad_shader_;
        bool initialized_ = false;
    };

} // namespace gs::visualizer