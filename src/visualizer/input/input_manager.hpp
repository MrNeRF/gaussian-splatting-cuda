#pragma once

#include "core/events.hpp"
#include "input/camera_controller.hpp"
#include "input/input_handler.hpp"
#include "internal/viewport.hpp"
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace gs::visualizer {

    class InputManager {
    public:
        using FileDropCallback = std::function<bool(const std::filesystem::path&, bool is_dataset)>;
        using GuiActiveCheck = std::function<bool()>;

        explicit InputManager(GLFWwindow* window, Viewport& viewport);
        ~InputManager();

        // Setup
        void initialize();
        void setupCallbacks(GuiActiveCheck gui_check, FileDropCallback file_drop);

        // Getters
        InputHandler* getInputHandler() { return input_handler_.get(); }
        CameraController* getCameraController() { return camera_controller_.get(); }

    private:
        GLFWwindow* window_;
        Viewport& viewport_;

        std::unique_ptr<InputHandler> input_handler_;
        std::unique_ptr<CameraController> camera_controller_;

        GuiActiveCheck gui_active_check_;
        FileDropCallback file_drop_callback_;

        // Handler IDs for cleanup
        std::vector<InputHandler::HandlerId> gui_handler_ids_;

        void setupInputHandlers();
        bool handleFileDrop(const InputHandler::FileDropEvent& event);
    };

} // namespace gs::visualizer
