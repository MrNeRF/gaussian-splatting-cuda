#pragma once

#include "visualizer/input_handler.hpp"
#include "visualizer/viewport.hpp"
#include <glm/glm.hpp>

namespace gs {

    class CameraController {
    public:
        explicit CameraController(Viewport& viewport) : viewport_(viewport) {}

        // Setup input handlers
        void connectToInputHandler(InputHandler& input_handler);

    private:
        // Input event handlers
        bool handleMouseButton(const InputHandler::MouseButtonEvent& event);
        bool handleMouseMove(const InputHandler::MouseMoveEvent& event);
        bool handleMouseScroll(const InputHandler::MouseScrollEvent& event);
        bool handleKey(const InputHandler::KeyEvent& event);

        Viewport& viewport_;
        InputHandler* input_handler_ = nullptr; // Store reference for key state queries

        // State
        bool is_panning_ = false;
        bool is_rotating_ = false;
        bool is_orbiting_ = false;
        bool is_enabled_ = true;
    };

} // namespace gs
