#include "input/input_manager.hpp"
#include <imgui.h>
#include <print>

namespace gs::visualizer {

    InputManager::InputManager(GLFWwindow* window, Viewport& viewport)
        : window_(window),
          viewport_(viewport) {
    }

    void InputManager::initialize() {
        // Create input handler
        input_handler_ = std::make_unique<InputHandler>(window_);

        // Create camera controller with viewport focus check
        camera_controller_ = std::make_unique<CameraController>(viewport_, viewport_focus_check_);
        camera_controller_->connectToInputHandler(*input_handler_);

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
        }
    }

    void InputManager::updateInputRouting() {
        // Check ImGui state
        bool imgui_wants_mouse = ImGui::GetIO().WantCaptureMouse;
        bool imgui_wants_keyboard = ImGui::GetIO().WantCaptureKeyboard;
        bool any_window_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);

        // Check if mouse is clicking in viewport
        bool mouse_clicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                             ImGui::IsMouseClicked(ImGuiMouseButton_Right) ||
                             ImGui::IsMouseClicked(ImGuiMouseButton_Middle);

        // If clicking and viewport has focus (from hover), force viewport input
        if (mouse_clicked && viewport_focus_check_ && viewport_focus_check_()) {
            // Force viewport to get input on this click
            input_handler_->setInputConsumer(InputHandler::InputConsumer::Viewport);

            // Clear ImGui's capture flags so it doesn't steal this click
            if (!any_window_hovered) {
                ImGui::GetIO().WantCaptureMouse = false;
            }
        } else if (imgui_wants_mouse || imgui_wants_keyboard || any_window_hovered) {
            input_handler_->setInputConsumer(InputHandler::InputConsumer::GUI);
        } else {
            // Default to viewport when GUI doesn't need input
            input_handler_->setInputConsumer(InputHandler::InputConsumer::Viewport);
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