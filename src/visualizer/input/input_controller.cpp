/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_controller.hpp"
#include "core/logger.hpp"
#include "tools/tool_base.hpp"
#include "tools/translation_gizmo_tool.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <format>
#include <imgui.h>

namespace gs::visualizer {
    InputController* InputController::instance_ = nullptr;

    InputController::InputController(GLFWwindow* window, Viewport& viewport)
        : window_(window),
          viewport_(viewport) {
        // Subscribe to GoToCamView events
        events::cmd::GoToCamView::when([this](const auto& e) {
            handleGoToCamView(e);
        });
        // Subscribe to WindowFocusLost to reset states
        events::internal::WindowFocusLost::when([this](const auto&) {
            drag_mode_ = DragMode::None;
            std::fill(std::begin(keys_wasd_), std::end(keys_wasd_), false);
        });

        LOG_DEBUG("InputController created");
    }

    InputController::~InputController() {
        if (instance_ == this) {
            instance_ = nullptr;
        }
    }

    void InputController::initialize() {
        // CRITICAL: This must be called AFTER ImGui_ImplGlfw_InitForOpenGL
        instance_ = this;

        // Store ImGui's callbacks so we can chain to them
        imgui_callbacks_.mouse_button = glfwSetMouseButtonCallback(window_, mouseButtonCallback);
        imgui_callbacks_.cursor_pos = glfwSetCursorPosCallback(window_, cursorPosCallback);
        imgui_callbacks_.scroll = glfwSetScrollCallback(window_, scrollCallback);
        imgui_callbacks_.key = glfwSetKeyCallback(window_, keyCallback);
        imgui_callbacks_.drop = glfwSetDropCallback(window_, dropCallback);
        imgui_callbacks_.focus = glfwSetWindowFocusCallback(window_, windowFocusCallback);

        // Get initial mouse position
        double x, y;
        glfwGetCursorPos(window_, &x, &y);
        last_mouse_pos_ = {x, y};

        // Initialize frame timer
        last_frame_time_ = std::chrono::high_resolution_clock::now();

        LOG_DEBUG("InputController initialized - callbacks set");
    }

    // Static callbacks - chain to ImGui then handle ourselves
    void InputController::mouseButtonCallback(GLFWwindow* w, int button, int action, int mods) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.mouse_button) {
            instance_->imgui_callbacks_.mouse_button(w, button, action, mods);
        }

        // Then handle for camera
        if (instance_) {
            double x, y;
            glfwGetCursorPos(w, &x, &y);
            instance_->handleMouseButton(button, action, x, y);
        }
    }

    void InputController::cursorPosCallback(GLFWwindow* w, double x, double y) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.cursor_pos) {
            instance_->imgui_callbacks_.cursor_pos(w, x, y);
        }

        // Then handle for camera
        if (instance_) {
            instance_->handleMouseMove(x, y);
        }
    }

    void InputController::scrollCallback(GLFWwindow* w, double xoff, double yoff) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.scroll) {
            instance_->imgui_callbacks_.scroll(w, xoff, yoff);
        }

        // Then handle for camera
        if (instance_) {
            instance_->handleScroll(xoff, yoff);
        }
    }

    void InputController::keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.key) {
            instance_->imgui_callbacks_.key(w, key, scancode, action, mods);
        }

        // Then handle for camera
        if (instance_) {
            instance_->handleKey(key, action, mods);
        }
    }

    void InputController::dropCallback(GLFWwindow* w, int count, const char** paths) {
        // Let ImGui handle first (though it probably doesn't use this)
        if (instance_ && instance_->imgui_callbacks_.drop) {
            instance_->imgui_callbacks_.drop(w, count, paths);
        }

        // Then handle file drops
        if (instance_) {
            std::vector<std::string> files(paths, paths + count);
            instance_->handleFileDrop(files);
        }
    }

    void InputController::windowFocusCallback(GLFWwindow* w, int focused) {
        // Let ImGui handle first
        if (instance_ && instance_->imgui_callbacks_.focus) {
            instance_->imgui_callbacks_.focus(w, focused);
        }

        // Reset states on focus loss
        if (!focused) {
            if (instance_) {
                instance_->drag_mode_ = DragMode::None;
                std::fill(std::begin(instance_->keys_wasd_),
                          std::end(instance_->keys_wasd_), false);
            }
            events::internal::WindowFocusLost{}.emit();
            LOG_DEBUG("Window lost focus - input states reset");
        } else {
            LOG_DEBUG("Window gained focus");
        }
    }

    // Core handlers
    void InputController::handleMouseButton(int button, int action, double x, double y) {
        // CHECK GIZMO FIRST - before any other input handling
        if (translation_gizmo_ && translation_gizmo_->isEnabled() && tool_context_) {
            if (translation_gizmo_->handleMouseButton(button, action, x, y, *tool_context_)) {
                // Gizmo consumed the event
                if (action == GLFW_PRESS) {
                    drag_mode_ = DragMode::Gizmo;
                    LOG_TRACE("Started gizmo drag");
                } else if (action == GLFW_RELEASE && drag_mode_ == DragMode::Gizmo) {
                    drag_mode_ = DragMode::None;
                    LOG_TRACE("Ended gizmo drag");
                }
                return; // Don't process camera controls
            }
        }

        if (action == GLFW_PRESS) {
            // Special handling for point cloud mode
            bool imgui_wants_mouse = ImGui::GetIO().WantCaptureMouse;
            if (point_cloud_mode_ && isInViewport(x, y) &&
                !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
                imgui_wants_mouse = false; // Override for point cloud mode
                LOG_TRACE("Point cloud mode - overriding ImGui mouse capture");
            }

            // Check if we should handle this
            if (imgui_wants_mouse || !isInViewport(x, y)) {
                return;
            }

            // Start camera interaction
            viewport_.camera.initScreenPos(glm::vec2(x, y));

            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                drag_mode_ = DragMode::Pan;
                LOG_TRACE("Started camera pan");
            } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                drag_mode_ = DragMode::Rotate;
                LOG_TRACE("Started camera rotate");
            } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
                drag_mode_ = DragMode::Orbit;
                float current_time = static_cast<float>(glfwGetTime());
                viewport_.camera.startRotateAroundCenter(glm::vec2(x, y), current_time);
                LOG_TRACE("Started camera orbit");
            }
        } else if (action == GLFW_RELEASE) {
            // Always handle our own releases if we were dragging
            bool was_dragging = false;
            if (button == GLFW_MOUSE_BUTTON_LEFT && drag_mode_ == DragMode::Pan) {
                drag_mode_ = DragMode::None;
                was_dragging = true;
                LOG_TRACE("Ended camera pan");
            } else if (button == GLFW_MOUSE_BUTTON_RIGHT && drag_mode_ == DragMode::Rotate) {
                drag_mode_ = DragMode::None;
                was_dragging = true;
                LOG_TRACE("Ended camera rotate");
            } else if (button == GLFW_MOUSE_BUTTON_MIDDLE && drag_mode_ == DragMode::Orbit) {
                viewport_.camera.endRotateAroundCenter();
                drag_mode_ = DragMode::None;
                was_dragging = true;
                LOG_TRACE("Ended camera orbit");
            }

            // Force publish on mouse release
            if (was_dragging) {
                events::ui::CameraMove{
                    .rotation = viewport_.getRotationMatrix(),
                    .translation = viewport_.getTranslation()}
                    .emit();
            }
        }
    }

    void InputController::handleMouseMove(double x, double y) {
        // Check gizmo first if not already in gizmo drag mode
        if (translation_gizmo_ && translation_gizmo_->isEnabled() && tool_context_) {
            if (drag_mode_ == DragMode::Gizmo ||
                translation_gizmo_->handleMouseMove(x, y, *tool_context_)) {
                // Gizmo is handling the mouse move
                last_mouse_pos_ = {x, y};
                return;
            }
        }

        glm::vec2 pos(x, y);

        // Always update last position
        glm::dvec2 current_pos{x, y};
        last_mouse_pos_ = current_pos;

        // Handle camera dragging
        if (drag_mode_ != DragMode::None && drag_mode_ != DragMode::Gizmo) {
            switch (drag_mode_) {
            case DragMode::Pan:
                viewport_.camera.translate(pos);
                break;
            case DragMode::Rotate:
                viewport_.camera.rotate(pos);
                break;
            case DragMode::Orbit: {
                float current_time = static_cast<float>(glfwGetTime());
                viewport_.camera.updateRotateAroundCenter(pos, current_time);
                break;
            }
            default:
                break;
            }
            publishCameraMove();
        }
    }

    void InputController::handleScroll([[maybe_unused]] double xoff, double yoff) {
        // Don't scroll if gizmo is active
        if (drag_mode_ == DragMode::Gizmo) {
            return;
        }

        if (!shouldCameraHandleInput())
            return;

        float delta = static_cast<float>(yoff);
        if (std::abs(delta) < 0.01f)
            return;

        if (key_r_pressed_) {
            viewport_.camera.rotate_roll(delta);
            LOG_TRACE("Camera roll: {}", delta);
        } else {
            viewport_.camera.zoom(delta);
            LOG_TRACE("Camera zoom: {}", delta);
        }

        publishCameraMove();
    }

    void InputController::handleKey(int key, int action, [[maybe_unused]] int mods) {
        // Track modifier keys (always, even if GUI has focus)
        if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
            key_ctrl_pressed_ = (action != GLFW_RELEASE);
        }
        if (key == GLFW_KEY_R) {
            key_r_pressed_ = (action != GLFW_RELEASE);
            // Reset gizmo position if R is pressed and gizmo is enabled
            if (action == GLFW_PRESS && translation_gizmo_ && translation_gizmo_->isEnabled()) {
                LOG_DEBUG("Reset key pressed with gizmo enabled");
                // This would need to be exposed by the gizmo tool
                // For now, let the gizmo handle it internally
            }
        }

        // Speed control works even when GUI has focus
        if (key_ctrl_pressed_ && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
                updateCameraSpeed(true);
                return;
            }
            if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
                updateCameraSpeed(false);
                return;
            }
        }

        // WASD only works when viewport has focus and gizmo isn't active
        if (!shouldCameraHandleInput() || drag_mode_ == DragMode::Gizmo)
            return;

        bool pressed = (action != GLFW_RELEASE);
        bool changed = false;

        switch (key) {
        case GLFW_KEY_W:
            keys_wasd_[0] = pressed;
            changed = true;
            break;
        case GLFW_KEY_A:
            keys_wasd_[1] = pressed;
            changed = true;
            break;
        case GLFW_KEY_S:
            keys_wasd_[2] = pressed;
            changed = true;
            break;
        case GLFW_KEY_D:
            keys_wasd_[3] = pressed;
            changed = true;
            break;
        }

        if (changed) {
            LOG_TRACE("WASD state changed - W:{} A:{} S:{} D:{}",
                      keys_wasd_[0], keys_wasd_[1], keys_wasd_[2], keys_wasd_[3]);
        }
    }

    void InputController::processWASDMovement() {
        // Calculate frame delta time
        auto now = std::chrono::high_resolution_clock::now();
        float delta_time = std::chrono::duration<float>(now - last_frame_time_).count();
        last_frame_time_ = now;

        // Clamp delta time to prevent huge jumps (60 FPS min)
        delta_time = std::min(delta_time, 1.0f / 60.0f);

        // Only process WASD if we should handle input and not dragging anything
        if (!shouldCameraHandleInput() || drag_mode_ != DragMode::None) {
            return;
        }

        bool any_movement = false;

        // Process each WASD key with frame-time based movement
        if (keys_wasd_[0]) { // W
            viewport_.camera.advance_forward(delta_time);
            any_movement = true;
        }
        if (keys_wasd_[1]) { // A
            viewport_.camera.advance_left(delta_time);
            any_movement = true;
        }
        if (keys_wasd_[2]) { // S
            viewport_.camera.advance_backward(delta_time);
            any_movement = true;
        }
        if (keys_wasd_[3]) { // D
            viewport_.camera.advance_right(delta_time);
            any_movement = true;
        }

        // Publish camera move if we moved
        if (any_movement) {
            publishCameraMove();
            LOG_TRACE("WASD movement - W:{} A:{} S:{} D:{}",
                      keys_wasd_[0], keys_wasd_[1], keys_wasd_[2], keys_wasd_[3]);
        }
    }

    void InputController::update(float delta_time) {
        // This catches cases where mouse release events are missed (e.g., outside window)
        if (drag_mode_ == DragMode::Orbit &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) != GLFW_PRESS) {
            viewport_.camera.endRotateAroundCenter();
            drag_mode_ = DragMode::None;
            LOG_TRACE("Orbit stopped - button released outside window");
        }

        if (drag_mode_ == DragMode::Rotate &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS) {
            drag_mode_ = DragMode::None;
            LOG_TRACE("Rotate stopped - button released outside window");
        }

        if (drag_mode_ == DragMode::Pan &&
            glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
            drag_mode_ = DragMode::None;
            LOG_TRACE("Pan stopped - button released outside window");
        }

        // Handle continuous WASD movement
        if (shouldCameraHandleInput() && drag_mode_ != DragMode::Gizmo) {
            if (keys_wasd_[0]) {
                viewport_.camera.advance_forward(delta_time);
            }
            if (keys_wasd_[1]) {
                viewport_.camera.advance_left(delta_time);
            }
            if (keys_wasd_[2]) {
                viewport_.camera.advance_backward(delta_time);
            }
            if (keys_wasd_[3]) {
                viewport_.camera.advance_right(delta_time);
            }
        }

        // Publish if moving (removed inertia check)
        bool moving = keys_wasd_[0] || keys_wasd_[1] || keys_wasd_[2] || keys_wasd_[3];
        if (moving) {
            publishCameraMove();
        }
    }

    void InputController::handleFileDrop(const std::vector<std::string>& paths) {
        LOG_DEBUG("Handling file drop with {} files", paths.size());

        std::vector<std::filesystem::path> ply_files;
        std::optional<std::filesystem::path> dataset_path;

        for (const auto& path_str : paths) {
            std::filesystem::path filepath(path_str);
            LOG_TRACE("Processing dropped file: {}", filepath.string());

            if (filepath.extension() == ".ply" || filepath.extension() == ".PLY") {
                ply_files.push_back(filepath);
            } else if (!dataset_path && std::filesystem::is_directory(filepath)) {
                // Check for dataset markers
                LOG_TRACE("Checking directory for dataset markers: {}", filepath.string());
                if (gs::loader::Loader::isDatasetPath(filepath)) {
                    dataset_path = filepath;
                    LOG_DEBUG("Dataset detected in dropped directory");
                }
            }
        }

        // Load PLY files
        for (const auto& ply : ply_files) {
            events::cmd::LoadFile{.path = ply, .is_dataset = false}.emit();
            LOG_INFO("Loading PLY via drag-and-drop: {}", ply.filename().string());
        }

        // Load dataset if found
        if (dataset_path) {
            events::cmd::LoadFile{.path = *dataset_path, .is_dataset = true}.emit();
            LOG_INFO("Loading dataset via drag-and-drop: {}", dataset_path->filename().string());
        }

        if (paths.size() == 1) {
            auto project_path = std::filesystem::path(paths[0]);
            if (project_path.extension() == gs::management::Project::EXTENSION) {
                events::cmd::LoadProject{.path = project_path}.emit();
                LOG_INFO("Loading LS Project via drag-and-drop: {}", project_path.filename().string());
            }
        }
    }

    void InputController::handleGoToCamView(const events::cmd::GoToCamView& event) {
        LOG_TIMER_TRACE("HandleGoToCamView");

        if (!training_manager_) {
            LOG_ERROR("GoToCamView: trainer_manager_ not initialized");
            return;
        }

        auto cam_data = training_manager_->getCamById(event.cam_id);
        if (!cam_data) {
            LOG_ERROR("Camera ID {} not found", event.cam_id);
            return;
        }

        LOG_DEBUG("Moving camera to view ID: {}", event.cam_id);

        // Transform from WorldToCam to CamToWorld
        glm::mat3 world_to_cam_R;
        auto R_accessor = cam_data->R().accessor<float, 2>();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                world_to_cam_R[j][i] = R_accessor[i][j];
            }
        }

        auto T_accessor = cam_data->T().accessor<float, 1>();
        glm::vec3 world_to_cam_T(T_accessor[0], T_accessor[1], T_accessor[2]);

        glm::mat3 cam_to_world_R = glm::transpose(world_to_cam_R);
        glm::vec3 cam_to_world_T = -cam_to_world_R * world_to_cam_T;

        viewport_.camera.R = cam_to_world_R;
        viewport_.camera.t = cam_to_world_T;

        // Update FOV
        float focal_x = cam_data->focal_x();
        float width = cam_data->image_width();
        if (focal_x > 0.0f && width > 0) {
            float fov_rad = 2.0f * std::atan(width / (2.0f * focal_x));
            float fov_deg = glm::degrees(fov_rad);
            LOG_TRACE("Setting FOV to {:.2f} degrees", fov_deg);
            events::ui::RenderSettingsChanged{
                .fov = fov_deg,
                .scaling_modifier = std::nullopt,
                .antialiasing = std::nullopt,
                .background_color = std::nullopt}
                .emit();
        }

        // Force immediate camera update
        events::ui::CameraMove{
            .rotation = viewport_.getRotationMatrix(),
            .translation = viewport_.getTranslation()}
            .emit();

        LOG_INFO("Camera moved to view: {} (ID: {})",
                 cam_data->image_name(), cam_data->uid());
    }

    // Helpers
    bool InputController::isInViewport(double x, double y) const {
        return x >= viewport_bounds_.x &&
               x < viewport_bounds_.x + viewport_bounds_.width &&
               y >= viewport_bounds_.y &&
               y < viewport_bounds_.y + viewport_bounds_.height;
    }

    bool InputController::shouldCameraHandleInput() const {
        // Don't handle if gizmo is active
        if (drag_mode_ == DragMode::Gizmo) {
            return false;
        }

        // Special handling for point cloud mode
        if (point_cloud_mode_) {
            double x, y;
            glfwGetCursorPos(window_, &x, &y);
            if (isInViewport(x, y) && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
                return true; // Force handle in point cloud mode
            }
        }

        return !ImGui::GetIO().WantCaptureKeyboard &&
               !ImGui::GetIO().WantCaptureMouse;
    }

    void InputController::updateCameraSpeed(bool increase) {
        if (increase) {
            viewport_.camera.increaseWasdSpeed();
        } else {
            viewport_.camera.decreaseWasdSpeed();
        }

        const float new_speed = viewport_.camera.getWasdSpeed();
        const float max_speed = viewport_.camera.getMaxWasdSpeed();

        LOG_DEBUG("Camera speed changed to: {:.3f} (max: {:.3f})", new_speed, max_speed);

        events::ui::SpeedChanged{
            .current_speed = new_speed,
            .max_speed = max_speed}
            .emit();
    }

    void InputController::publishCameraMove() {
        auto now = std::chrono::steady_clock::now();
        if (now - last_camera_publish_ >= camera_publish_interval_) {
            events::ui::CameraMove{
                .rotation = viewport_.getRotationMatrix(),
                .translation = viewport_.getTranslation()}
                .emit();
            last_camera_publish_ = now;
        }
    }
} // namespace gs::visualizer