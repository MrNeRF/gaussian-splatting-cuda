#include "input/camera_controller.hpp"
#include "core/events.hpp"

// clang-format off
// CRITICAL: GLAD must be included before GLFW to avoid OpenGL header conflicts
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <cmath>

namespace gs {

    CameraController::CameraController(Viewport& viewport, std::function<bool()> viewport_focus_check)
        : viewport_(viewport),
          viewport_focus_check_(viewport_focus_check) {}

    CameraController::~CameraController() {
        // No cleanup needed - input handler manages callbacks now
    }

    void CameraController::connectToInputHandler(InputHandler& input_handler) {
        // Store reference to input handler for checking key states
        input_handler_ = &input_handler;
    }

    bool CameraController::isViewportFocused() const {
        if (viewport_focus_check_) {
            return viewport_focus_check_();
        }
        return true; // Default to focused if no check function
    }

    bool CameraController::isPositionInViewport(double x, double y) const {
        if (position_check_callback_) {
            return position_check_callback_(x, y);
        }
        // Fallback to regular focus check
        return isViewportFocused();
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

    void CameraController::handleMouseButton(const InputHandler::MouseButtonEvent& event) {
        if (!is_enabled_)
            return;

        if (event.action == GLFW_PRESS) {
            // Check if the mouse position is in the viewport
            bool in_viewport = isPositionInViewport(event.position.x, event.position.y);

            if (in_viewport) {
                // Initialize camera position
                viewport_.camera.initScreenPos(glm::vec2(event.position));

                // Start the appropriate interaction
                if (event.button == GLFW_MOUSE_BUTTON_LEFT) {
                    is_panning_ = true;
                } else if (event.button == GLFW_MOUSE_BUTTON_RIGHT) {
                    is_rotating_ = true;
                } else if (event.button == GLFW_MOUSE_BUTTON_MIDDLE) {
                    is_orbiting_ = true;
                }
            }
        } else if (event.action == GLFW_RELEASE) {
            // Always handle our own releases
            if (event.button == GLFW_MOUSE_BUTTON_LEFT && is_panning_) {
                is_panning_ = false;
                // Force publish on mouse release to ensure final position is sent
                events::ui::CameraMove{
                    .rotation = viewport_.getRotationMatrix(),
                    .translation = viewport_.getTranslation()}
                    .emit();
            } else if (event.button == GLFW_MOUSE_BUTTON_RIGHT && is_rotating_) {
                is_rotating_ = false;
                // Force publish on mouse release
                events::ui::CameraMove{
                    .rotation = viewport_.getRotationMatrix(),
                    .translation = viewport_.getTranslation()}
                    .emit();
            } else if (event.button == GLFW_MOUSE_BUTTON_MIDDLE && is_orbiting_) {
                is_orbiting_ = false;
                // Force publish on mouse release
                events::ui::CameraMove{
                    .rotation = viewport_.getRotationMatrix(),
                    .translation = viewport_.getTranslation()}
                    .emit();
            }
        }
    }

    void CameraController::handleMouseMove(const InputHandler::MouseMoveEvent& event) {
        if (!is_enabled_)
            return;

        // Always handle mouse movement if we're actively dragging
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
        }
    }

    void CameraController::handleMouseScroll(const InputHandler::MouseScrollEvent& event) {
        if (!is_enabled_ || !isViewportFocused())
            return;

        float delta = static_cast<float>(event.yoffset);
        if (std::abs(delta) < 1.0e-2f)
            return;

        // Check if R key is pressed for roll using the input handler
        if (input_handler_ && input_handler_->isKeyPressed(GLFW_KEY_R)) {
            viewport_.camera.rotate_roll(delta);
        } else {
            viewport_.camera.zoom(delta);
        }

        publishCameraChanged();
    }

    bool CameraController::handleSpeedChange([[maybe_unused]] const InputHandler::KeyEvent& event) {
        // Handle speed control first (Ctrl + Plus/Minus)
        if (input_handler_ && (input_handler_->isKeyPressed(GLFW_KEY_LEFT_CONTROL) ||
                               input_handler_->isKeyPressed(GLFW_KEY_RIGHT_CONTROL))) {

            if (input_handler_->isKeyPressed(GLFW_KEY_EQUAL) ||
                input_handler_->isKeyPressed(GLFW_KEY_KP_ADD)) {
                // Increase speed (Ctrl + '+' or Ctrl + '=')
                viewport_.camera.increaseWasdSpeed();

                // Emit speed change event
                events::ui::SpeedChanged{
                    .current_speed = viewport_.camera.getWasdSpeed(),
                    .max_speed = viewport_.camera.getMaxWasdSpeed()}
                    .emit();

                return true;
            }
            if (input_handler_->isKeyPressed(GLFW_KEY_MINUS) ||
                input_handler_->isKeyPressed(GLFW_KEY_KP_SUBTRACT)) {
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
        return false;
    }

    bool CameraController::handleWasd([[maybe_unused]] const InputHandler::KeyEvent& event) {
        // Only handle WASD if viewport is focused
        if (!isViewportFocused())
            return false;

        const float ADVANCE_RATE = 1.0f;
        float advance_rate = ADVANCE_RATE;
        bool camera_changed = false;

        if (input_handler_->isKeyPressed(GLFW_KEY_W)) {
            viewport_.camera.advance_forward(advance_rate);
            camera_changed = true;
        }

        if (input_handler_->isKeyPressed(GLFW_KEY_S)) {
            viewport_.camera.advance_backward(advance_rate);
            camera_changed = true;
        }

        if (input_handler_->isKeyPressed(GLFW_KEY_A)) {
            viewport_.camera.advance_left(advance_rate);
            camera_changed = true;
        }

        if (input_handler_->isKeyPressed(GLFW_KEY_D)) {
            viewport_.camera.advance_right(advance_rate);
            camera_changed = true;
        }

        if (camera_changed) {
            publishCameraChanged();
            return true;
        }

        return false;
    }

    void CameraController::handleKey(const InputHandler::KeyEvent& event) {
        if (!is_enabled_)
            return;

        // Speed changes work even when viewport isn't focused (global shortcuts)
        handleSpeedChange(event);

        // WASD only works when viewport is focused
        handleWasd(event);
    }

    void CameraController::resetStates() {
        is_panning_ = false;
        is_rotating_ = false;
        is_orbiting_ = false;
    }

} // namespace gs