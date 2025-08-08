#pragma once

#include "core/events.hpp"
#include "training/training_manager.hpp"
#include "input/camera_controller.hpp"
#include "input/input_handler.hpp"
#include "internal/viewport.hpp"
#include <filesystem>
#include <functional>
#include <memory>

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
        void setViewportFocusCheck(std::function<bool()> focus_check);
        void setPositionCheck(std::function<bool(double, double)> check);

        // Update input routing based on focus
        void updateInputRouting();

        // Getters
        InputHandler* getInputHandler() { return input_handler_.get(); }
        CameraController* getCameraController() { return camera_controller_.get(); }

        void setTrainingManager(std::shared_ptr<const TrainerManager> training_manager){ trainer_manager_ = training_manager;};

    private:
        GLFWwindow* window_;
        Viewport& viewport_;

        std::unique_ptr<InputHandler> input_handler_;
        std::unique_ptr<CameraController> camera_controller_;

        std::shared_ptr<const TrainerManager> trainer_manager_;

        GuiActiveCheck gui_active_check_;
        FileDropCallback file_drop_callback_;
        std::function<bool()> viewport_focus_check_;
        std::function<bool(double, double)> position_check_;

        void setupInputHandlers();
        void handleFileDrop(const InputHandler::FileDropEvent& event);

        // Event handlers
        void setupEventHandlers();
        void handleGoToCamView(const events::cmd::GoToCamView& event);
    };

} // namespace gs::visualizer