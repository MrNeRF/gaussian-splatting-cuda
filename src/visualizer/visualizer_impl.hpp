#pragma once

#include "core/error_handler.hpp"
#include "core/main_loop.hpp"
#include "core/memory_monitor.hpp"
#include "core/parameters.hpp"
#include "gui/gui_manager.hpp"
#include "input/input_controller.hpp"
#include "internal/viewport.hpp"
#include "project/project.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "tools/tool_manager.hpp"
#include "training/training_manager.hpp"
#include "visualizer/visualizer.hpp"
#include "window/window_manager.hpp"
#include <memory>
#include <string>

// Forward declaration for GLFW
struct GLFWwindow;

namespace gs {
    class CommandProcessor;
    class SceneManager;
} // namespace gs

namespace gs::visualizer {
    class DataLoadingService;

    class VisualizerImpl : public Visualizer {
    public:
        explicit VisualizerImpl(const ViewerOptions& options);
        ~VisualizerImpl() override;

        void run() override;
        void setParameters(const param::TrainingParameters& params) override;
        std::expected<void, std::string> loadPLY(const std::filesystem::path& path) override;
        std::expected<void, std::string> loadDataset(const std::filesystem::path& path) override;
        void clearScene() override;

        // open project file and attach it to viewer
        bool openProject(const std::filesystem::path& path) override;
        bool closeProject(const std::filesystem::path& path) override;
        std::shared_ptr<gs::management::Project> getProject() override;
        // load project content to viewer
        void LoadProject();

        // Getters for GUI (delegating to state manager)
        Trainer* getTrainer() const { return trainer_manager_->getTrainer(); }

        // Component access
        TrainerManager* getTrainerManager() { return trainer_manager_.get(); }
        SceneManager* getSceneManager() { return scene_manager_.get(); }
        ::GLFWwindow* getWindow() const { return window_manager_->getWindow(); }
        ToolManager* getToolManager() { return tool_manager_.get(); }
        RenderingManager* getRenderingManager() { return rendering_manager_.get(); }
        const Viewport& getViewport() const { return viewport_; }

        // FPS monitoring
        [[nodiscard]] float getCurrentFPS() const {
            return rendering_manager_ ? rendering_manager_->getCurrentFPS() : 0.0f;
        }

        [[nodiscard]] float getAverageFPS() const {
            return rendering_manager_ ? rendering_manager_->getAverageFPS() : 0.0f;
        }

        // VSync control
        void setVSync(bool enabled) {
            if (window_manager_) {
                window_manager_->setVSync(enabled);
            }
        }

        [[nodiscard]] bool getVSyncEnabled() const {
            return window_manager_ ? window_manager_->getVSync() : true;
        }

        // Antialiasing state
        bool isAntiAliasingEnabled() const {
            return rendering_manager_ ? rendering_manager_->getSettings().antialiasing : false;
        }

        // Tool helpers
        std::shared_ptr<gs::rendering::IBoundingBox> getCropBox() const;
        std::shared_ptr<const gs::rendering::ICoordinateAxes> getAxes() const;
        std::shared_ptr<const geometry::EuclideanTransform> getWorldToUser() const;

        std::shared_ptr<TrainerManager> trainer_manager_;

        // GUI manager
        std::unique_ptr<gui::GuiManager> gui_manager_;
        friend class gui::GuiManager;
        friend class ToolManager;

    private:
        // Main loop callbacks
        bool initialize();
        void update();
        void render();
        void shutdown();

        // Event system
        void setupEventHandlers();
        void setupComponentConnections();

        // Options
        ViewerOptions options_;

        // Core components
        Viewport viewport_;
        std::unique_ptr<WindowManager> window_manager_;
        std::unique_ptr<InputController> input_controller_;
        std::unique_ptr<RenderingManager> rendering_manager_;
        std::unique_ptr<SceneManager> scene_manager_;
        std::unique_ptr<CommandProcessor> command_processor_;
        std::unique_ptr<DataLoadingService> data_loader_;
        std::unique_ptr<MainLoop> main_loop_;
        std::unique_ptr<ToolManager> tool_manager_;

        // Support components
        std::unique_ptr<ErrorHandler> error_handler_;
        std::unique_ptr<MemoryMonitor> memory_monitor_;

        // State
        bool gui_initialized_ = false;
        // Project
        std::shared_ptr<gs::management::Project> project_ = nullptr;
    };

} // namespace gs::visualizer