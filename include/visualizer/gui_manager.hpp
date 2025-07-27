#pragma once

#include "core/trainer.hpp"
#include "visualizer/event_bus.hpp"
#include "visualizer/events.hpp"
#include "visualizer/viewer_notifier.hpp"
#include <chrono>
#include <filesystem>
#include <functional>
#include <imgui.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gs {

    // Forward declarations
    class GSViewer;

    namespace gui {

        // Scripting console component
        class ScriptingConsole {
        public:
            ScriptingConsole();

            void clearLog();
            void addLog(const char* fmt, ...);
            void executeCommand(const std::string& command);
            void render(bool* p_open);
            void setExecutor(std::function<std::string(const std::string&)> executor);

        private:
            static int textEditCallbackStub(ImGuiInputTextCallbackData* data);
            int textEditCallback(ImGuiInputTextCallbackData* data);

            std::vector<std::string> history_;
            std::vector<std::string> output_buffer_;
            char input_buffer_[1024] = "";
            int history_pos_ = -1;
            bool scroll_to_bottom_ = false;
            bool reclaim_focus_ = false;
            size_t max_output_lines_ = 1000;
            std::function<std::string(const std::string&)> execute_callback_;
        };

        // File browser component
        class FileBrowser {
        public:
            FileBrowser();

            void render(bool* p_open);
            void setOnFileSelected(std::function<void(const std::filesystem::path&, bool)> callback);
            void setCurrentPath(const std::filesystem::path& path);

        private:
            std::string current_path_;
            std::string selected_file_;
            std::function<void(const std::filesystem::path&, bool)> on_file_selected_;
        };

        // Camera controls window
        class CameraControlsWindow {
        public:
            void render(bool* p_open);
        };

        // Training controls panel
        class TrainingControlsPanel {
        public:
            struct State {
                bool manual_start_triggered = false;
                bool training_started = false;
                bool save_in_progress = false;
                std::chrono::steady_clock::time_point save_start_time;
            };

            void render(Trainer* trainer, State& state, std::shared_ptr<ViewerNotifier> notifier);

        private:
            void renderStartButton(State& state);
            void renderRunningControls(Trainer* trainer, State& state);
            void renderStatus(Trainer* trainer, State& state);
        };

        // Main GUI manager
        class GuiManager {
        public:
            GuiManager(GSViewer* viewer, std::shared_ptr<EventBus> event_bus);
            ~GuiManager();

            void init();
            void beginFrame();
            void endFrame();
            void render();
            void shutdown();

            // Window visibility controls
            void showFileBrowser(bool show = true);
            void showScriptingConsole(bool show = true);
            void showCameraControls(bool show = true);

            // GUI state
            bool isAnyWindowActive() const { return any_window_active_; }

            // Callbacks
            void setScriptExecutor(std::function<std::string(const std::string&)> executor);
            void setFileSelectedCallback(std::function<void(const std::filesystem::path&, bool)> callback);

            // Console access
            void addConsoleLog(const char* fmt, ...);

            // Get viewer for internal use
            GSViewer* viewer_;

            // Helper to publish events - MOVED TO PUBLIC
            template <typename EventType>
            void publish(const EventType& event) {
                if (event_bus_) {
                    event_bus_->publish(event);
                }
            }

        private:
            void renderMainPanel();
            void renderModeStatus();
            void renderRenderingSettings();
            void renderProgressInfo();

            // Event system
            std::shared_ptr<EventBus> event_bus_;
            std::vector<size_t> event_handler_ids_;

            // Event handlers
            void setupEventHandlers();
            void handleSceneLoaded(const SceneLoadedEvent& event);
            void handleSceneCleared(const SceneClearedEvent& event);
            void handleTrainingStarted(const TrainingStartedEvent& event);
            void handleTrainingProgress(const TrainingProgressEvent& event);
            void handleTrainingPaused(const TrainingPausedEvent& event);
            void handleTrainingResumed(const TrainingResumedEvent& event);
            void handleTrainingCompleted(const TrainingCompletedEvent& event);
            void handleLogMessage(const LogMessageEvent& event);

            // Components
            std::unique_ptr<ScriptingConsole> scripting_console_;
            std::unique_ptr<FileBrowser> file_browser_;
            std::unique_ptr<CameraControlsWindow> camera_controls_;
            std::unique_ptr<TrainingControlsPanel> training_controls_;

            // Window states
            bool show_main_panel_ = true;
            bool show_file_browser_ = false;
            bool show_scripting_console_ = false;
            bool show_camera_controls_ = false;
            bool any_window_active_ = false;

            // ImGui settings
            ImGuiWindowFlags window_flags_ = 0;

            // Training controls state
            TrainingControlsPanel::State training_state_;
        };

    } // namespace gui
} // namespace gs