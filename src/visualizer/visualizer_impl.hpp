#pragma once

#include "core/error_handler.hpp"
#include "core/main_loop.hpp"
#include "core/memory_monitor.hpp"
#include "core/parameters.hpp"
#include "core/viewer_state_manager.hpp"
#include "project/project.hpp"
#include "gui/gui_manager.hpp"
#include "input/input_manager.hpp"
#include "internal/viewport.hpp"
#include "rendering/render_bounding_box.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene.hpp"
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
}

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

        bool openProject(const std::filesystem::path& path) override;
        bool closeProject(const std::filesystem::path& path) override;
        std::shared_ptr<gs::management::Project> getProject(const std::filesystem::path& path) override;

        // Getters for GUI (delegating to state manager)
        ViewerMode getCurrentMode() const { return state_manager_->getCurrentMode(); }
        Trainer* getTrainer() const { return trainer_manager_->getTrainer(); }
        std::shared_ptr<TrainingInfo> getTrainingInfo() const { return state_manager_->getTrainingInfo(); }
        std::shared_ptr<RenderingConfig> getRenderingConfig() const { return state_manager_->getRenderingConfig(); }
        const std::filesystem::path& getCurrentPLYPath() const { return state_manager_->getCurrentPLYPath(); }
        const std::filesystem::path& getCurrentDatasetPath() const { return state_manager_->getCurrentDatasetPath(); }
        TrainerManager* getTrainerManager() { return trainer_manager_.get(); }
        SceneManager* getSceneManager() { return scene_manager_.get(); }
        ::GLFWwindow* getWindow() const { return window_manager_->getWindow(); }
        ToolManager* getToolManager() { return tool_manager_.get(); }
        RenderingManager* getRenderingManager() { return rendering_manager_.get(); }
        const Viewport& getViewport() const { return viewport_; } // Add viewport getter

        // Add FPS monitoring methods
        [[nodiscard]] float getCurrentFPS() const {
            return rendering_manager_ ? rendering_manager_->getCurrentFPS() : 0.0f;
        }

        [[nodiscard]] float getAverageFPS() const {
            return rendering_manager_ ? rendering_manager_->getAverageFPS() : 0.0f;
        }

        // Add VSync control methods
        void setVSync(bool enabled) {
            if (window_manager_) {
                window_manager_->setVSync(enabled);
            }
        }

        [[nodiscard]] bool getVSyncEnabled() const {
            return window_manager_ ? window_manager_->getVSync() : true;
        }

        // Compatibility method for crop box
        std::shared_ptr<RenderBoundingBox> getCropBox() const;
        std::shared_ptr<const RenderCoordinateAxes> getAxes() const;
        std::shared_ptr<const geometry::EuclideanTransform> getWorldToUser() const;

        // GUI needs these for compatibility
        std::shared_ptr<TrainingInfo> info_;
        std::shared_ptr<RenderingConfig> config_;
        bool anti_aliasing_ = false; // Temporary for compatibility

        // Scene management (temporarily public for compatibility)
        std::unique_ptr<Scene> scene_;
        std::shared_ptr<TrainerManager> trainer_manager_;

        // GUI manager
        std::unique_ptr<gui::GuiManager> gui_manager_;
        friend class gui::GuiManager;
        friend class ToolManager; // Add friend declaration for ToolManager

    private:
        // Main loop callbacks
        bool initialize();
        void update();
        void render();
        void forceRender();
        void shutdown();

        // Event system
        void setupEventHandlers();
        void setupComponentConnections();

        // Options
        ViewerOptions options_;

        // Core components
        Viewport viewport_;
        std::unique_ptr<WindowManager> window_manager_;
        std::unique_ptr<InputManager> input_manager_;
        std::unique_ptr<RenderingManager> rendering_manager_;
        std::unique_ptr<SceneManager> scene_manager_;
        std::unique_ptr<ViewerStateManager> state_manager_;
        std::unique_ptr<CommandProcessor> command_processor_;
        std::unique_ptr<DataLoadingService> data_loader_;
        std::unique_ptr<MainLoop> main_loop_;
        std::unique_ptr<ToolManager> tool_manager_;

        // Support components
        std::unique_ptr<ErrorHandler> error_handler_;
        std::unique_ptr<MemoryMonitor> memory_monitor_;
        // State
        bool gui_initialized_ = false;
        //Project
        std::shared_ptr<gs::management::Project> project_ = nullptr;
    };

} // namespace gs::visualizer