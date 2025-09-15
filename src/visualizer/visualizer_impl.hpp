/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/main_loop.hpp"
#include "core/parameters.hpp"
#include "gui/gui_manager.hpp"
#include "input/input_controller.hpp"
#include "internal/viewport.hpp"
#include "project/project.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "tools/tool_base.hpp"
#include "training/training_manager.hpp"
#include "visualizer/visualizer.hpp"
#include "window/window_manager.hpp"
#include <memory>
#include <string>

// Forward declaration for GLFW
struct GLFWwindow;

namespace gs {
    class SceneManager;
} // namespace gs

namespace gs::visualizer {
    class DataLoadingService;

    namespace tools {
        class TranslationGizmoTool;
    }

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
        bool closeProject(const std::filesystem::path& path = {}) override;
        std::shared_ptr<gs::management::Project> getProject() override;
        void attachProject(std::shared_ptr<gs::management::Project> _project) override;
        // load project content to viewer
        bool LoadProject();
        void LoadProjectPlys();

        // Getters for GUI (delegating to state manager)
        gs::training::Trainer* getTrainer() const { return trainer_manager_->getTrainer(); }

        // Component access
        TrainerManager* getTrainerManager() { return trainer_manager_.get(); }
        SceneManager* getSceneManager() { return scene_manager_.get(); }
        ::GLFWwindow* getWindow() const { return window_manager_->getWindow(); }
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

        tools::TranslationGizmoTool* getTranslationGizmoTool() {
            return translation_gizmo_tool_.get();
        }

        const tools::TranslationGizmoTool* getTranslationGizmoTool() const {
            return translation_gizmo_tool_.get();
        }

        std::shared_ptr<TrainerManager> trainer_manager_;

        // GUI manager
        std::unique_ptr<gui::GuiManager> gui_manager_;
        friend class gui::GuiManager;

        // Allow ToolContext to access GUI manager for logging
        friend class ToolContext;

    private:
        // Main loop callbacks
        bool initialize();
        void update();
        void render();
        void shutdown();
        bool allowclose();

        // Event system
        void setupEventHandlers();
        void setupComponentConnections();
        void handleLoadProjectCommand(const events::cmd::LoadProject& cmd);
        void handleTrainingCompleted(const events::state::TrainingCompleted& event);
        void handleLoadFileCommand(const events::cmd::LoadFile& cmd);
        void handleSaveProject(const events::cmd::SaveProject& cmd);

        // Tool initialization
        void initializeTools();

        // Options
        ViewerOptions options_;

        // Core components
        Viewport viewport_;
        std::unique_ptr<WindowManager> window_manager_;
        std::unique_ptr<InputController> input_controller_;
        std::unique_ptr<RenderingManager> rendering_manager_;
        std::unique_ptr<SceneManager> scene_manager_;
        std::unique_ptr<DataLoadingService> data_loader_;
        std::unique_ptr<MainLoop> main_loop_;

        // Tools
        std::shared_ptr<tools::TranslationGizmoTool> translation_gizmo_tool_;
        std::unique_ptr<ToolContext> tool_context_;

        // State tracking
        bool window_initialized_ = false;
        bool gui_initialized_ = false;
        bool tools_initialized_ = false; // Added this member!
        // Project
        std::shared_ptr<gs::management::Project> project_ = nullptr;
        void updateProjectOnModules();
    };

} // namespace gs::visualizer