#pragma once

#include "framerate_controller.hpp"
#include "internal/viewport.hpp"
#include "rendering/rendering.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>

namespace gs {
    class SceneManager;
} // namespace gs

namespace gs::visualizer {

    // Forward declaration
    class BackgroundTool;

    struct RenderSettings {
        // Core rendering settings
        float fov = 60.0f;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        bool show_crop_box = false;
        bool use_crop_box = false;
        bool show_coord_axes = false;
        bool show_grid = true;
        int grid_plane = 1; // Default to XZ plane
        float grid_opacity = 0.5f;
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

        // Check if initialized
        bool isInitialized() const { return initialized_; }

        // Main render function
        void renderFrame(const RenderContext& context, SceneManager* scene_manager);

        // Mark that rendering is needed
        void markDirty();

        // Settings management
        void updateSettings(const RenderSettings& settings);
        RenderSettings getSettings() const;

        // Direct accessors
        float getFovDegrees() const;
        float getScalingModifier() const;
        void setFov(float f);
        void setScalingModifier(float s);

        // FPS monitoring
        float getCurrentFPS() const { return framerate_controller_.getCurrentFPS(); }
        float getAverageFPS() const { return framerate_controller_.getAverageFPS(); }

        // Access to rendering engine (for initialization only)
        gs::rendering::RenderingEngine* getRenderingEngine();

    private:
        void doFullRender(const RenderContext& context, SceneManager* scene_manager, const SplatData* model);
        void renderOverlays(const RenderContext& context);
        void setupEventHandlers();

        // Core components
        std::unique_ptr<gs::rendering::RenderingEngine> engine_;
        FramerateController framerate_controller_;

        // Minimal state
        std::atomic<bool> needs_render_{true};
        gs::rendering::RenderResult cached_result_;
        size_t last_model_ptr_ = 0;
        std::chrono::steady_clock::time_point last_training_render_;

        // Settings
        RenderSettings settings_;
        mutable std::mutex settings_mutex_;

        bool initialized_ = false;
    };

} // namespace gs::visualizer