#pragma once

#include "core/events.hpp"
#include "input/input_handler.hpp"
#include "internal/viewport.hpp"
#include <chrono>
#include <vector>

namespace gs {

    class CameraController {
    public:
        explicit CameraController(Viewport& viewport) : viewport_(viewport) {}
        ~CameraController();

        // Setup input handlers
        void connectToInputHandler(InputHandler& input_handler);

    private:
        // Input event handlers
        bool handleMouseButton(const InputHandler::MouseButtonEvent& event);
        bool handleMouseMove(const InputHandler::MouseMoveEvent& event);
        bool handleMouseScroll(const InputHandler::MouseScrollEvent& event);
        bool handleKey(const InputHandler::KeyEvent& event);

        // Publish camera changed event
        void publishCameraChanged();

        Viewport& viewport_;
        InputHandler* input_handler_ = nullptr; // Store reference for key state queries

        // Handler IDs for cleanup
        std::vector<InputHandler::HandlerId> handler_ids_;

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
