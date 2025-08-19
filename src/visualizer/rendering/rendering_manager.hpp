#pragma once

#include "core/events.hpp"
#include "framerate_controller.hpp"
#include "internal/viewport.hpp"
#include "rendering/rendering.hpp"
#include <memory>

namespace gs {
    namespace rendering {
        class RenderBoundingBox;
        class RenderCoordinateAxes;
    } // namespace rendering
    namespace geometry {
        class EuclideanTransform;
    }
    class SceneManager;
} // namespace gs

namespace gs::visualizer {

    // Forward declaration
    class BackgroundTool;

    struct RenderSettings {
        float fov = 60.0f;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        bool show_crop_box = false;
        bool use_crop_box = false;
        bool show_coord_axes = false;
        bool show_grid = true;
        int grid_plane = 1; // Default to XZ plane
        float grid_opacity = 0.5f;
        bool adaptive_frame_rate = true;
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;
    };

    struct ViewportRegion {
        float x, y, width, height;
    };

    class RenderingManager {
    public:
        struct RenderContext {
            const Viewport& viewport;
            const RenderSettings& settings;
            const gs::rendering::IBoundingBox* crop_box;
            const gs::rendering::ICoordinateAxes* coord_axes;
            const geometry::EuclideanTransform* world_to_user;
            const ViewportRegion* viewport_region = nullptr;
            bool has_focus = false;
            const BackgroundTool* background_tool = nullptr;
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
        gs::rendering::RenderingEngine* getRenderingEngine() { return engine_.get(); }

    private:
        void drawSceneFrame(const RenderContext& context, SceneManager* scene_manager, bool skip_render);
        void drawFocusIndicator(const RenderContext& context);
        void drawOverlays(const RenderContext& context);
        bool hasCamChanged(const class Viewport& current_viewport);
        bool hasSceneChanged(const RenderContext& context);
        void setupEventHandlers();

        RenderSettings settings_;
        std::unique_ptr<gs::rendering::RenderingEngine> engine_;
        bool initialized_ = false;

        // Framerate control
        FramerateController framerate_controller_;
        std::unique_ptr<Viewport> prev_viewport_state_;
        float prev_fov_ = 0;
        std::unique_ptr<geometry::EuclideanTransform> prev_world_to_usr_inv_;
        glm::vec3 prev_background_color_;
        glm::ivec2 prev_render_size_;
        gs::rendering::RenderResult prev_result_;

        bool prev_point_cloud_mode_ = false;
        float prev_voxel_size_ = 0.01f;

        // Scene loading tracking - for frame control
        bool scene_just_loaded_ = false;
        event::HandlerId scene_loaded_handler_id_ = 0;
        event::HandlerId grid_settings_handler_id_ = 0;
        event::HandlerId scene_changed_handler_id_ = 0;
    };

} // namespace gs::visualizer