#pragma once

#include "core/events.hpp"
#include "input/input_handler.hpp"
#include "internal/viewport.hpp"
#include <chrono>
#include <functional>
#include <vector>

namespace gs {

    class CameraController {
    public:
        explicit CameraController(Viewport& viewport,
                                  std::function<bool()> viewport_focus_check = nullptr);
        ~CameraController();

        // Setup input handlers
        void connectToInputHandler(InputHandler& input_handler);

        // Handle input events
        void handleMouseButton(const InputHandler::MouseButtonEvent& event);
        void handleMouseMove(const InputHandler::MouseMoveEvent& event);
        void handleMouseScroll(const InputHandler::MouseScrollEvent& event);
        void handleKey(const InputHandler::KeyEvent& event);

    private:
        // Check if viewport is focused
        bool isViewportFocused() const;

        // Input event handlers
        bool handleSpeedChange(const InputHandler::KeyEvent& event);
        bool handleWasd(const InputHandler::KeyEvent& event);

        // Publish camera changed event
        void publishCameraChanged();

        Viewport& viewport_;
        InputHandler* input_handler_ = nullptr; // Store reference for key state queries
        std::function<bool()> viewport_focus_check_;

        // State
        bool is_panning_ = false;
        bool is_rotating_ = false;
        bool is_orbiting_ = false;
        bool is_enabled_ = true;

        // Throttling for camera events
        std::chrono::steady_clock::time_point last_camera_publish_time_;
        static constexpr std::chrono::milliseconds camera_publish_interval_ms_{100};
    };

} // namespace gs