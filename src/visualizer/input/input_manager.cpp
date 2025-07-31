#include "input/input_manager.hpp"
#include <algorithm>
#include <imgui.h>
#include <print>

namespace gs::visualizer {

    InputManager::InputManager(GLFWwindow* window, Viewport& viewport)
        : window_(window),
          viewport_(viewport) {
    }

    InputManager::~InputManager() {
        // Clean up our handlers
        if (input_handler_) {
            for (auto id : gui_handler_ids_) {
                input_handler_->removeHandler(id);
            }
        }
    }

    void InputManager::initialize() {
        // Create input handler
        input_handler_ = std::make_unique<InputHandler>(window_);

        // Create camera controller
        camera_controller_ = std::make_unique<CameraController>(viewport_);
        camera_controller_->connectToInputHandler(*input_handler_);
    }

    void InputManager::setupCallbacks(GuiActiveCheck gui_check, FileDropCallback file_drop) {
        gui_active_check_ = gui_check;
        file_drop_callback_ = file_drop;
        setupInputHandlers();
    }

    void InputManager::setupInputHandlers() {
        if (!input_handler_)
            return;

        // Clear existing GUI handlers
        for (auto id : gui_handler_ids_) {
            input_handler_->removeHandler(id);
        }
        gui_handler_ids_.clear();

        // GUI gets highest priority for all mouse input
        gui_handler_ids_.push_back(
            input_handler_->addMouseButtonHandler(
                [this](const InputHandler::MouseButtonEvent&) {
                    return ImGui::GetIO().WantCaptureMouse;
                },
                InputPriority::GUI));

        gui_handler_ids_.push_back(
            input_handler_->addMouseMoveHandler(
                [this](const InputHandler::MouseMoveEvent&) {
                    return ImGui::GetIO().WantCaptureMouse;
                },
                InputPriority::GUI));

        gui_handler_ids_.push_back(
            input_handler_->addMouseScrollHandler(
                [this](const InputHandler::MouseScrollEvent&) {
                    return ImGui::GetIO().WantCaptureMouse;
                },
                InputPriority::GUI));

        // GUI gets highest priority for keyboard input
        gui_handler_ids_.push_back(
            input_handler_->addKeyHandler(
                [this](const InputHandler::KeyEvent&) {
                    return ImGui::GetIO().WantCaptureKeyboard;
                },
                InputPriority::GUI));

        // File drop handler - high priority
        gui_handler_ids_.push_back(
            input_handler_->addFileDropHandler(
                [this](const InputHandler::FileDropEvent& event) {
                    return handleFileDrop(event);
                },
                InputPriority::System));
    }

    bool InputManager::handleFileDrop(const InputHandler::FileDropEvent& event) {
        if (!file_drop_callback_)
            return false;

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
                    return true;
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
                        return true;
                    }
                }
            }
        }
        return false;
    }

} // namespace gs::visualizer
