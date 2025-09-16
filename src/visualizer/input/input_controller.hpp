/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "input/input_types.hpp"
#include "internal/viewport.hpp"
#include "training/training_manager.hpp"
#include <GLFW/glfw3.h>
#include <chrono>
#include <glm/glm.hpp>
#include <memory>

namespace gs::visualizer {

    // Forward declarations
    namespace tools {
        class TranslationGizmoTool;
    }
    class ToolContext;
    class RenderingManager;

    class InputController {
    public:
        InputController(GLFWwindow* window, Viewport& viewport);
        ~InputController();

        // Setup - MUST be called AFTER ImGui is initialized!
        void initialize();

        // Set training manager for camera view commands
        void setTrainingManager(std::shared_ptr<const TrainerManager> tm) {
            training_manager_ = tm;
        }

        // Set translation gizmo tool
        void setTranslationGizmoTool(std::shared_ptr<tools::TranslationGizmoTool> tool) {
            translation_gizmo_ = tool;
        }

        // Set tool context for gizmo
        void setToolContext(ToolContext* context) {
            tool_context_ = context;
        }

        // Set rendering manager for split view
        void setRenderingManager(RenderingManager* rm) {
            rendering_manager_ = rm;
        }

        // Called every frame by GUI manager to update viewport bounds
        void updateViewportBounds(float x, float y, float w, float h) {
            viewport_bounds_ = {x, y, w, h};
        }

        // Set special input modes
        void setPointCloudMode(bool enabled) {
            point_cloud_mode_ = enabled;
        }

        // Update function for continuous input (WASD movement and inertia)
        void update(float delta_time);

    private:
        // Store original ImGui callbacks so we can chain
        struct {
            GLFWmousebuttonfun mouse_button = nullptr;
            GLFWcursorposfun cursor_pos = nullptr;
            GLFWscrollfun scroll = nullptr;
            GLFWkeyfun key = nullptr;
            GLFWdropfun drop = nullptr;
            GLFWwindowfocusfun focus = nullptr;
        } imgui_callbacks_;

        // Our callbacks that chain to ImGui
        static void mouseButtonCallback(GLFWwindow* w, int button, int action, int mods);
        static void cursorPosCallback(GLFWwindow* w, double x, double y);
        static void scrollCallback(GLFWwindow* w, double xoff, double yoff);
        static void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods);
        static void dropCallback(GLFWwindow* w, int count, const char** paths);
        static void windowFocusCallback(GLFWwindow* w, int focused);

        // Internal handlers
        void handleMouseButton(int button, int action, double x, double y);
        void handleMouseMove(double x, double y);
        void handleScroll(double xoff, double yoff);
        void handleKey(int key, int action, int mods);
        void handleFileDrop(const std::vector<std::string>& paths);
        void handleGoToCamView(const events::cmd::GoToCamView& event);

        // WASD processing with proper frame timing
        void processWASDMovement();

        // Helpers
        bool isInViewport(double x, double y) const;
        bool shouldCameraHandleInput() const;
        void updateCameraSpeed(bool increase);
        void publishCameraMove();
        bool isNearSplitter(double x) const;

        // Core state
        GLFWwindow* window_;
        Viewport& viewport_;
        std::shared_ptr<const TrainerManager> training_manager_;
        RenderingManager* rendering_manager_ = nullptr;

        // Tool support
        std::shared_ptr<tools::TranslationGizmoTool> translation_gizmo_;
        ToolContext* tool_context_ = nullptr;

        // Viewport bounds for focus detection
        struct {
            float x, y, width, height;
        } viewport_bounds_{0, 0, 1920, 1080};

        // Camera state
        enum class DragMode {
            None,
            Pan,
            Rotate,
            Orbit,
            Gizmo,
            Splitter
        };
        DragMode drag_mode_ = DragMode::None;
        glm::dvec2 last_mouse_pos_{0, 0};
        float splitter_start_pos_ = 0.5f;
        double splitter_start_x_ = 0.0;
        bool gimbal_locked = false;

        // Key states (only what we actually need)
        bool key_r_pressed_ = false;
        bool key_ctrl_pressed_ = false;
        bool keys_wasd_[6] = {false, false, false, false, false, false}; // W,A,S,D,Q,E

        // Special modes
        bool point_cloud_mode_ = false;

        // Throttling for camera events
        std::chrono::steady_clock::time_point last_camera_publish_;
        static constexpr auto camera_publish_interval_ = std::chrono::milliseconds(100);

        // Frame timing for WASD movement
        std::chrono::high_resolution_clock::time_point last_frame_time_;

        // Cursor state tracking
        enum class CursorType {
            Default,
            Resize,
            Hand
        };
        CursorType current_cursor_ = CursorType::Default;
        GLFWcursor* resize_cursor_ = nullptr;
        GLFWcursor* hand_cursor_ = nullptr;

        // Camera frustum interaction
        int last_camview = -1;
        int hovered_camera_id_ = -1;
        int last_clicked_camera_id_ = -1;
        std::chrono::steady_clock::time_point last_click_time_;
        glm::dvec2 last_click_pos_{0, 0};
        static constexpr double DOUBLE_CLICK_TIME = 0.3;     // seconds
        static constexpr double DOUBLE_CLICK_DISTANCE = 5.0; // pixels

        // Static instance for callbacks
        static InputController* instance_;
    };

} // namespace gs::visualizer