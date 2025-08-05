#include "input/input_manager.hpp"
#include <imgui.h>
#include <print>

namespace gs::visualizer {

    InputManager::InputManager(GLFWwindow* window, Viewport& viewport)
        : window_(window),
          viewport_(viewport) {
    }

    InputManager::~InputManager() {
        // Cleanup handled automatically by unique_ptr
    }

    void InputManager::initialize() {
        // Create input handler
        input_handler_ = std::make_unique<InputHandler>(window_);

        // Create camera controller with viewport focus check
        camera_controller_ = std::make_unique<CameraController>(viewport_, viewport_focus_check_);
        camera_controller_->connectToInputHandler(*input_handler_);

        // Pass position check to camera controller
        if (position_check_) {
            camera_controller_->setPositionCheckCallback(position_check_);
        }

        setupInputHandlers();
    }

    void InputManager::setupCallbacks(GuiActiveCheck gui_check, FileDropCallback file_drop) {
        gui_active_check_ = gui_check;
        file_drop_callback_ = file_drop;
    }

    void InputManager::setViewportFocusCheck(std::function<bool()> focus_check) {
        viewport_focus_check_ = focus_check;

        // Update camera controller if it exists
        if (camera_controller_) {
            // Recreate camera controller with new focus check
            camera_controller_ = std::make_unique<CameraController>(viewport_, viewport_focus_check_);
            camera_controller_->connectToInputHandler(*input_handler_);

            // Reapply position check if it exists
            if (position_check_) {
                camera_controller_->setPositionCheckCallback(position_check_);
            }
        }
    }

    void InputManager::setPositionCheck(std::function<bool(double, double)> check) {
        position_check_ = check;

        // Pass to input handler for viewport detection
        if (input_handler_) {
            input_handler_->setViewportCheckCallback(check);
        }

        // Also pass to camera controller
        if (camera_controller_) {
            camera_controller_->setPositionCheckCallback(check);
        }
    }

    void InputManager::updateInputRouting() {
        // Simple focus-based routing for keyboard and scroll
        bool viewport_has_focus = viewport_focus_check_ ? viewport_focus_check_() : false;

        if (viewport_has_focus && !ImGui::GetIO().WantCaptureKeyboard) {
            input_handler_->setInputConsumer(InputHandler::InputConsumer::Viewport);
        } else {
            input_handler_->setInputConsumer(InputHandler::InputConsumer::GUI);
        }
    }

    void InputManager::setupInputHandlers() {
        if (!input_handler_)
            return;

        // Set viewport callbacks
        input_handler_->setViewportCallbacks(
            [this](const InputHandler::MouseButtonEvent& event) {
                if (camera_controller_) {
                    camera_controller_->handleMouseButton(event);
                }
            },
            [this](const InputHandler::MouseMoveEvent& event) {
                if (camera_controller_) {
                    camera_controller_->handleMouseMove(event);
                }
            },
            [this](const InputHandler::MouseScrollEvent& event) {
                if (camera_controller_) {
                    camera_controller_->handleMouseScroll(event);
                }
            },
            [this](const InputHandler::KeyEvent& event) {
                if (camera_controller_) {
                    camera_controller_->handleKey(event);
                }
            });

        // GUI just uses ImGui's input handling, no special callbacks needed

        // File drop handler
        input_handler_->setFileDropCallback(
            [this](const InputHandler::FileDropEvent& event) {
                handleFileDrop(event);
            });
    }

    void InputManager::handleFileDrop(const InputHandler::FileDropEvent& event) {
        if (!file_drop_callback_)
            return;

        // Process each dropped file
        for (const auto& path_str : event.paths) {
            std::filesystem::path filepath(path_str);

            // Check if it's a PLY file
            if (filepath.extension() == ".ply" || filepath.extension() == ".PLY") {
                std::println("Dropped PLY file: {}", filepath.string());

                if (file_drop_callback_(filepath, false)) {
                    // Log the action
                    events::notify::Log{
                        .level = events::notify::Log::Level::Info,
                        .message = std::format("Loaded PLY file via drag-and-drop: {}",
                                               filepath.filename().string()),
                        .source = "InputManager"}
                        .emit();
                    return;
                }
            }

            if (std::filesystem::is_directory(filepath)) {
                // Check if it's a dataset directory
                bool is_colmap_dataset = false;
                bool is_transforms_dataset = false;

                // Check for COLMAP dataset structure
                if (std::filesystem::exists(filepath / "sparse" / "0" / "cameras.bin") ||
                    std::filesystem::exists(filepath / "sparse" / "cameras.bin")) {
                    is_colmap_dataset = true;
                }

                // Check for transforms dataset
                if (std::filesystem::exists(filepath / "transforms.json") ||
                    std::filesystem::exists(filepath / "transforms_train.json")) {
                    is_transforms_dataset = true;
                }

                if (is_colmap_dataset || is_transforms_dataset) {
                    std::println("Dropped dataset directory: {}", filepath.string());

                    if (file_drop_callback_(filepath, true)) {
                        // Log the action
                        events::notify::Log{
                            .level = events::notify::Log::Level::Info,
                            .message = std::format("Loaded {} dataset via drag-and-drop: {}",
                                                   is_colmap_dataset ? "COLMAP" : "Transforms",
                                                   filepath.filename().string()),
                            .source = "InputManager"}
                            .emit();
                        return;
                    }
                }
            }
        }
    }

} // namespace gs::visualizer