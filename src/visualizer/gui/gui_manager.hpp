#pragma once

#include "core/events.hpp"
#include "gui/ui_context.hpp"
#include <filesystem>
#include <imgui.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace gs {
    namespace visualizer {
        class VisualizerImpl;
    }

    namespace gui {
        class ScriptingConsole;
        class FileBrowser;
        class ScenePanel;

        class GuiManager {
        public:
            GuiManager(visualizer::VisualizerImpl* viewer);
            ~GuiManager();

            // Lifecycle
            void init();
            void shutdown();
            void render();

            // State queries
            bool wantsInput() const;
            bool isAnyWindowActive() const;
            bool showCropBox() const;
            bool useCropBox() const;

            // Window visibility
            void showWindow(const std::string& name, bool show = true);
            void toggleWindow(const std::string& name);

            // Console access for logging
            void addConsoleLog(const char* fmt, ...);

            // Missing methods that visualizer_impl expects
            void setScriptExecutor(std::function<std::string(const std::string&)> executor);
            void setFileSelectedCallback(std::function<void(const std::filesystem::path&, bool)> callback);
            void showScriptingConsole(bool show = true) { window_states_["console"] = show; }

        private:
            void setupEventHandlers();
            void applyDefaultStyle();

            // Core dependencies
            visualizer::VisualizerImpl* viewer_;

            // Owned components
            std::unique_ptr<ScriptingConsole> console_;
            std::unique_ptr<FileBrowser> file_browser_;
            std::unique_ptr<ScenePanel> scene_panel_;

            // UI state only
            std::unordered_map<std::string, bool> window_states_;
            bool show_main_panel_ = true;
            // Speed overlay state
            bool speed_overlay_visible_ = false;
            std::chrono::steady_clock::time_point speed_overlay_start_time_;
            std::chrono::milliseconds speed_overlay_duration_;
            float current_speed_;
            float max_speed_;

            // Method declarations to add to your GuiManager class:
            void renderSpeedOverlay();
            void showSpeedOverlay(float current_speed, float max_speed);
        };
    } // namespace gui
} // namespace gs
