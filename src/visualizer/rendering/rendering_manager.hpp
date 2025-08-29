/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

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

    struct RenderSettings {
        // Core rendering settings
        float fov = 60.0f;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;

        // Crop box
        bool show_crop_box = false;
        bool use_crop_box = false;
        glm::vec3 crop_min = glm::vec3(-1.0f, -1.0f, -1.0f);
        glm::vec3 crop_max = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 crop_color = glm::vec3(1.0f, 1.0f, 0.0f);
        float crop_line_width = 2.0f;
        geometry::EuclideanTransform crop_transform;

        // Background
        glm::vec3 background_color = glm::vec3(0.0f, 0.0f, 0.0f);

        // Coordinate axes
        bool show_coord_axes = false;
        float axes_size = 2.0f;
        std::array<bool, 3> axes_visibility = {true, true, true};

        // World transform
        geometry::EuclideanTransform world_transform;

        // Grid
        bool show_grid = true;
        int grid_plane = 1;
        float grid_opacity = 0.5f;

        // Point cloud
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;

        // Translation gizmo
        bool show_translation_gizmo = false;
        float gizmo_scale = 1.0f;

        bool gut = false;
    };

    struct ViewportRegion {
        float x, y, width, height;
    };

    class RenderingManager {
    public:
        struct RenderContext {
            const Viewport& viewport;
            const RenderSettings& settings;
            const ViewportRegion* viewport_region = nullptr;
            bool has_focus = false;
        };

        RenderingManager();
        ~RenderingManager();

        // Initialize rendering resources
        void initialize();
        bool isInitialized() const { return initialized_; }

        // Set initial viewport size (must be called before initialize())
        void setInitialViewportSize(const glm::ivec2& size) {
            initial_viewport_size_ = size;
        }

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

        // State tracking
        std::atomic<bool> needs_render_{true};
        gs::rendering::RenderResult cached_result_;
        size_t last_model_ptr_ = 0;
        glm::ivec2 last_render_size_{0, 0};
        std::chrono::steady_clock::time_point last_training_render_;

        // Settings
        RenderSettings settings_;
        mutable std::mutex settings_mutex_;

        bool initialized_ = false;
        glm::ivec2 initial_viewport_size_{1280, 720}; // Default fallback
    };

} // namespace gs::visualizer