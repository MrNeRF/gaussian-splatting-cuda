#pragma once

#include "rendering/framerate_controller.hpp"
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

        // Framerate control
        void updateFramerateSettings(const FramerateSettings& settings) { framerate_controller_.updateSettings(settings); }
        const FramerateSettings& getFramerateSettings() const { return framerate_controller_.getSettings(); }
        float getCurrentFPS() const { return framerate_controller_.getCurrentFPS(); }
        float getAverageFPS() const { return framerate_controller_.getAverageFPS(); }
        bool isPerformanceCritical() const { return framerate_controller_.isPerformanceCritical(); }
        void resetFramerateController() { framerate_controller_.reset(); }

        // Get shader for external use (crop box rendering)
        std::shared_ptr<Shader> getQuadShader() const { return quad_shader_; }

    private:
        void initializeShaders();
        void drawSceneFrame(const RenderContext& context, SceneManager* scene_manager, bool skip_render);
        void drawFocusIndicator(const RenderContext& context);
        void drawCropBox(const RenderContext& context);
        void drawCoordAxes(const RenderContext& context);
        bool hasViewportChanged(const Viewport& current_viewport) const;

        RenderSettings settings_;
        std::shared_ptr<ScreenQuadRenderer> screen_renderer_;
        std::shared_ptr<Shader> quad_shader_;
        bool initialized_ = false;

        // Framerate control
        FramerateController framerate_controller_;
        mutable bool last_viewport_changed_ = true;
        mutable Viewport last_viewport_state_;
        RenderingPipeline::RenderResult last_result_;
    };

} // namespace gs::visualizer