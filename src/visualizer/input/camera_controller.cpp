#include "input/camera_controller.hpp"
#include "core/events.hpp"

// clang-format off
// CRITICAL: GLAD must be included before GLFW to avoid OpenGL header conflicts
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <cmath>

namespace gs {

    CameraController::~CameraController() {
        // Clean up our handlers
        if (input_handler_) {
            for (auto id : handler_ids_) {
                input_handler_->removeHandler(id);
            }
        }
    }

    void CameraController::connectToInputHandler(InputHandler& input_handler) {
        // Store reference to input handler for checking key states
        input_handler_ = &input_handler;

        // Register all handlers with Camera priority
        handler_ids_.push_back(
            input_handler.addMouseButtonHandler(
                [this](const InputHandler::MouseButtonEvent& event) {
                    return handleMouseButton(event);
                },
                InputPriority::Camera));

        handler_ids_.push_back(
            input_handler.addMouseMoveHandler(
                [this](const InputHandler::MouseMoveEvent& event) {
                    return handleMouseMove(event);
                },
                InputPriority::Camera));

        handler_ids_.push_back(
            input_handler.addMouseScrollHandler(
                [this](const InputHandler::MouseScrollEvent& event) {
                    return handleMouseScroll(event);
                },
                InputPriority::Camera));

        handler_ids_.push_back(
            input_handler.addKeyHandler(
                [this](const InputHandler::KeyEvent& event) {
                    return handleKey(event);
                },
                InputPriority::Camera));
    }

    void CameraController::publishCameraChanged() {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_camera_publish_time_) >= camera_publish_interval_ms_) {
            events::ui::CameraMove{
                .rotation = viewport_.getRotationMatrix(),
                .translation = viewport_.getTranslation()}
                .emit();
            last_camera_publish_time_ = now;
        }
    }

    bool CameraController::handleMouseButton(const InputHandler::MouseButtonEvent& event) {
        if (!is_enabled_)
            return false;

        if (event.action == GLFW_PRESS) {
            viewport_.camera.initScreenPos(glm::vec2(event.position));

            if (event.button == GLFW_MOUSE_BUTTON_LEFT) {
                is_panning_ = true;
                return true;
            } else if (event.button == GLFW_MOUSE_BUTTON_RIGHT) {
                is_rotating_ = true;
                return true;
            } else if (event.button == GLFW_MOUSE_BUTTON_MIDDLE) {
                is_orbiting_ = true;
                return true;
            }
        } else if (event.action == GLFW_RELEASE) {
            if (event.button == GLFW_MOUSE_BUTTON_LEFT && is_panning_) {
                is_panning_ = false;
                // Force publish on mouse release to ensure final position is sent
                events::ui::CameraMove{
                    .rotation = viewport_.getRotationMatrix(),
                    .translation = viewport_.getTranslation()}
                    .emit();
                return true;
            } else if (event.button == GLFW_MOUSE_BUTTON_RIGHT && is_rotating_) {
                is_rotating_ = false;
                // Force publish on mouse release
                events::ui::CameraMove{
                    .rotation = viewport_.getRotationMatrix(),
                    .translation = viewport_.getTranslation()}
                    .emit();
                return true;
            } else if (event.button == GLFW_MOUSE_BUTTON_MIDDLE && is_orbiting_) {
                is_orbiting_ = false;
                // Force publish on mouse release
                events::ui::CameraMove{
                    .rotation = viewport_.getRotationMatrix(),
                    .translation = viewport_.getTranslation()}
                    .emit();
                return true;
            }
        }

        return false;
    }

    bool CameraController::handleMouseMove(const InputHandler::MouseMoveEvent& event) {
        if (!is_enabled_)
            return false;

        glm::vec2 current_pos(event.position);
        bool camera_changed = false;

        if (is_panning_) {
            viewport_.camera.translate(current_pos);
            camera_changed = true;
        } else if (is_rotating_) {
            viewport_.camera.rotate(current_pos);
            camera_changed = true;
        } else if (is_orbiting_) {
            viewport_.camera.rotate_around_center(current_pos);
            camera_changed = true;
        }

        if (camera_changed) {
            publishCameraChanged();
            return true;
        }

        return false;
    }

    bool CameraController::handleMouseScroll(const InputHandler::MouseScrollEvent& event) {
        if (!is_enabled_)
            return false;

        float delta = static_cast<float>(event.yoffset);
        if (std::abs(delta) < 1.0e-2f)
            return false;

        // Check if R key is pressed for roll using the input handler
        if (input_handler_ && input_handler_->isKeyPressed(GLFW_KEY_R)) {
            viewport_.camera.rotate_roll(delta);
        } else {
            viewport_.camera.zoom(delta);
        }

        publishCameraChanged();
        return true;
    }

    bool CameraController::handleKey(const InputHandler::KeyEvent& event) {
        if (!is_enabled_)
            return false;

        const float ADVANCE_RATE = 1.0f;
        const float ADVANCE_RATE_FINE_TUNE = 0.3f;

        // Handle speed control first (Ctrl + Plus/Minus)
        if (input_handler_ && (input_handler_->isKeyPressed(GLFW_KEY_LEFT_CONTROL) ||
                               input_handler_->isKeyPressed(GLFW_KEY_RIGHT_CONTROL))) {

            if (event.action == GLFW_PRESS) {
                if (event.key == GLFW_KEY_EQUAL || event.key == GLFW_KEY_KP_ADD) {
                    // Increase speed (Ctrl + '+' or Ctrl + '=')
                    viewport_.camera.increaseWasdSpeed();

                    // Emit speed change event
                    events::ui::SpeedChanged{
                        .current_speed = viewport_.camera.getWasdSpeed(),
                        .max_speed = viewport_.camera.getMaxWasdSpeed()}
                        .emit();

                    return true;
                } else if (event.key == GLFW_KEY_MINUS || event.key == GLFW_KEY_KP_SUBTRACT) {
                    // Decrease speed (Ctrl + '-')
                    viewport_.camera.decreaseWasdSpeed();

                    // Emit speed change event
                    events::ui::SpeedChanged{
                        .current_speed = viewport_.camera.getWasdSpeed(),
                        .max_speed = viewport_.camera.getMaxWasdSpeed()}
                        .emit();

                    return true;
                }
            }
            // If Ctrl is held but it's not +/-, don't process WASD movement
            return false;
        }

        // Handle WASD movement (only if Ctrl is not pressed)
        float advance_rate = 0.0f;
        if (event.action == GLFW_PRESS) {
            advance_rate = ADVANCE_RATE_FINE_TUNE;
        } else if (event.action == GLFW_REPEAT) {
            advance_rate = ADVANCE_RATE;
        } else {
            return false;
        }

        bool camera_changed = false;
        switch (event.key) {
        case GLFW_KEY_W:
            viewport_.camera.advance_forward(advance_rate);
            camera_changed = true;
            break;
        case GLFW_KEY_S:
            viewport_.camera.advance_backward(advance_rate);
            camera_changed = true;
            break;
        case GLFW_KEY_A:
            viewport_.camera.advance_left(advance_rate);
            camera_changed = true;
            break;
        case GLFW_KEY_D:
            viewport_.camera.advance_right(advance_rate);
            camera_changed = true;
            break;
        default:
            return false;
        }

        if (camera_changed) {
            publishCameraChanged();
            return true;
        }

        return false;
    }

} // namespace gs