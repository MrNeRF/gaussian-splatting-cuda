#include <algorithm>
#include <format>
#include <imgui.h>
#include <print>

#include "core/logger.hpp"
#include "input/input_controller.hpp"

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
            std::println("[InputController] Window lost focus");
        }
    }

    // Core handlers
    void InputController::handleMouseButton(int button, int action, double x, double y) {
        if (action == GLFW_PRESS) {
            // Special handling for point cloud mode
            bool imgui_wants_mouse = ImGui::GetIO().WantCaptureMouse;
            if (point_cloud_mode_ && isInViewport(x, y) &&
                !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
                imgui_wants_mouse = false; // Override for point cloud mode
            }

            // Check if we should handle this
            if (imgui_wants_mouse || !isInViewport(x, y)) {
                return;
            }

            // Start camera interaction
            viewport_.camera.initScreenPos(glm::vec2(x, y));

            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                drag_mode_ = DragMode::Pan;
            } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                drag_mode_ = DragMode::Rotate;
            } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
                drag_mode_ = DragMode::Orbit;
            }
        } else if (action == GLFW_RELEASE) {
            // Always handle our own releases if we were dragging
            bool was_dragging = false;

            if (button == GLFW_MOUSE_BUTTON_LEFT && drag_mode_ == DragMode::Pan) {
                drag_mode_ = DragMode::None;
                was_dragging = true;
            } else if (button == GLFW_MOUSE_BUTTON_RIGHT && drag_mode_ == DragMode::Rotate) {
                drag_mode_ = DragMode::None;
                was_dragging = true;
            } else if (button == GLFW_MOUSE_BUTTON_MIDDLE && drag_mode_ == DragMode::Orbit) {
                drag_mode_ = DragMode::None;
                was_dragging = true;
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
        glm::vec2 pos(x, y);

        // Always update last position
        glm::dvec2 current_pos{x, y};
        last_mouse_pos_ = current_pos;

        // Handle camera dragging
        if (drag_mode_ != DragMode::None) {
            switch (drag_mode_) {
            case DragMode::Pan:
                viewport_.camera.translate(pos);
                break;
            case DragMode::Rotate:
                viewport_.camera.rotate(pos);
                break;
            case DragMode::Orbit:
                viewport_.camera.rotate_around_center(pos);
                break;
            default:
                break;
            }

            publishCameraMove();
        }
    }

    void InputController::handleScroll([[maybe_unused]] double xoff, double yoff) {
        if (!shouldCameraHandleInput())
            return;

        float delta = static_cast<float>(yoff);
        if (std::abs(delta) < 0.01f)
            return;

        if (key_r_pressed_) {
            viewport_.camera.rotate_roll(delta);
        } else {
            viewport_.camera.zoom(delta);
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

        // WASD only works when viewport has focus
        if (!shouldCameraHandleInput())
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
            // Apply movement
            const float rate = 1.0f;
            if (keys_wasd_[0])
                viewport_.camera.advance_forward(rate);
            if (keys_wasd_[1])
                viewport_.camera.advance_left(rate);
            if (keys_wasd_[2])
                viewport_.camera.advance_backward(rate);
            if (keys_wasd_[3])
                viewport_.camera.advance_right(rate);

            if (keys_wasd_[0] || keys_wasd_[1] || keys_wasd_[2] || keys_wasd_[3]) {
                publishCameraMove();
            }
        }
    }

    void InputController::handleFileDrop(const std::vector<std::string>& paths) {
        std::vector<std::filesystem::path> ply_files;
        std::optional<std::filesystem::path> dataset_path;

        for (const auto& path_str : paths) {
            std::filesystem::path filepath(path_str);

            if (filepath.extension() == ".ply" || filepath.extension() == ".PLY") {
                ply_files.push_back(filepath);
            } else if (!dataset_path && std::filesystem::is_directory(filepath)) {
                // Check for dataset markers
                if (std::filesystem::exists(filepath / "sparse" / "0" / "cameras.bin") ||
                    std::filesystem::exists(filepath / "sparse" / "cameras.bin") ||
                    std::filesystem::exists(filepath / "transforms.json") ||
                    std::filesystem::exists(filepath / "transforms_train.json")) {
                    dataset_path = filepath;
                }
            }
        }

        // Load PLY files
        for (const auto& ply : ply_files) {
            events::cmd::LoadFile{.path = ply, .is_dataset = false}.emit();

            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = std::format("Loaded PLY via drag-and-drop: {}",
                                       ply.filename().string()),
                .source = "InputController"}
                .emit();
        }

        // Load dataset if found
        if (dataset_path) {
            events::cmd::LoadFile{.path = *dataset_path, .is_dataset = true}.emit();
            LOG_INFO("Loading dataset Project via drag-and-drop: {}", dataset_path->filename().string());
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
        if (!training_manager_) {
            std::cerr << "handleGoToCamView: trainer_manager_ not initialized\n";
            return;
        }

        auto cam_data = training_manager_->getCamById(event.cam_id);
        if (!cam_data) {
            std::cerr << "Camera ID " << event.cam_id << " not found\n";
            return;
        }

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

        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("Camera moved to view: {} (ID: {})",
                                   cam_data->image_name(), cam_data->uid()),
            .source = "CameraController"}
            .emit();
    }

    // Helpers
    bool InputController::isInViewport(double x, double y) const {
        return x >= viewport_bounds_.x &&
               x < viewport_bounds_.x + viewport_bounds_.width &&
               y >= viewport_bounds_.y &&
               y < viewport_bounds_.y + viewport_bounds_.height;
    }

    bool InputController::shouldCameraHandleInput() const {
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

        events::ui::SpeedChanged{
            .current_speed = viewport_.camera.getWasdSpeed(),
            .max_speed = viewport_.camera.getMaxWasdSpeed()}
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
