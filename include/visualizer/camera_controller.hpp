#pragma once

#include "visualizer/viewport.hpp"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace gs {

    class CameraController {
    public:
        explicit CameraController(Viewport& viewport) : viewport_(viewport) {}

        // Handle all camera-related input in one place
        void handleMouseButton(int button, int action, double x, double y) {
            if (action == GLFW_PRESS) {
                mouse_pos_ = glm::vec2(x, y);
                viewport_.camera.initScreenPos(mouse_pos_);

                if (button == GLFW_MOUSE_BUTTON_LEFT)
                    is_panning_ = true;
                else if (button == GLFW_MOUSE_BUTTON_RIGHT)
                    is_rotating_ = true;
                else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
                    is_orbiting_ = true;
            } else if (action == GLFW_RELEASE) {
                if (button == GLFW_MOUSE_BUTTON_LEFT)
                    is_panning_ = false;
                else if (button == GLFW_MOUSE_BUTTON_RIGHT)
                    is_rotating_ = false;
                else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
                    is_orbiting_ = false;
            }
        }

        void handleMouseMove(double x, double y) {
            glm::vec2 current_pos(x, y);

            if (is_panning_) {
                viewport_.camera.translate(current_pos);
            } else if (is_rotating_) {
                viewport_.camera.rotate(current_pos);
            } else if (is_orbiting_) {
                viewport_.camera.rotate_around_center(current_pos);
            }

            mouse_pos_ = current_pos;
        }

        void handleScroll(double yoffset, bool roll_modifier) {
            float delta = static_cast<float>(yoffset);
            if (std::abs(delta) < 1.0e-2f)
                return;

            if (roll_modifier) {
                viewport_.camera.rotate_roll(delta);
            } else {
                viewport_.camera.zoom(delta);
            }
        }

        void handleKeyboard(int key, float advance_rate) {
            switch (key) {
            case GLFW_KEY_W: viewport_.camera.advance_forward(advance_rate); break;
            case GLFW_KEY_S: viewport_.camera.advance_backward(advance_rate); break;
            case GLFW_KEY_A: viewport_.camera.advance_left(advance_rate); break;
            case GLFW_KEY_D: viewport_.camera.advance_right(advance_rate); break;
            }
        }

    private:
        Viewport& viewport_;
        glm::vec2 mouse_pos_{0.0f};
        bool is_panning_ = false;
        bool is_rotating_ = false;
        bool is_orbiting_ = false;
    };

} // namespace gs
