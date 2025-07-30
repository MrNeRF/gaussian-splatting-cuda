#pragma once

#include "core/event_bus.hpp"
#include "input/input_handler.hpp"
#include "internal/viewport.hpp"
#include <chrono>
#include <memory>

namespace gs {

    class CameraController {
    public:
        explicit CameraController(Viewport& viewport) : viewport_(viewport) {}

        // Setup input handlers
        void connectToInputHandler(InputHandler& input_handler);

        // Set event bus for publishing camera events
        void setEventBus(std::shared_ptr<EventBus> event_bus) { event_bus_ = event_bus; }

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
        std::shared_ptr<EventBus> event_bus_;

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