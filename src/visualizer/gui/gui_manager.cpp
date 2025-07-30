#include "gui/gui_manager.hpp"
#include "core/events.hpp"
#include "gui/panels/crop_box_panel.hpp"
#include "gui/panels/main_panel.hpp"
#include "gui/panels/scene_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/windows/camera_controls.hpp"
#include "gui/windows/file_browser.hpp"
#include "gui/windows/scripting_console.hpp"
#include "visualizer_impl.hpp"

#include <GLFW/glfw3.h>
#include <cstdarg>
#include <format>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace gs::gui {

    GuiManager::GuiManager(visualizer::VisualizerImpl* viewer, std::shared_ptr<EventBus> event_bus)
        : viewer_(viewer),
          event_bus_(event_bus) {

        // Create components
        console_ = std::make_unique<ScriptingConsole>();
        file_browser_ = std::make_unique<FileBrowser>();
        scene_panel_ = std::make_unique<ScenePanel>(*event_bus);

        // Initialize window states
        window_states_["console"] = false;
        window_states_["file_browser"] = false;
        window_states_["camera_controls"] = false;
        window_states_["scene_panel"] = true;

        setupEventHandlers();
    }

    GuiManager::~GuiManager() {
        // Event handlers are automatically cleaned up when event bus is destroyed
    }

    void GuiManager::init() {
        // ImGui initialization
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        io.ConfigWindowsMoveFromTitleBarOnly = true;

        // Platform/Renderer initialization
        ImGui_ImplGlfw_InitForOpenGL(viewer_->getWindow(), true);
        ImGui_ImplOpenGL3_Init("#version 430");

        // Load fonts
        std::string font_path = std::string(PROJECT_ROOT_PATH) +
                                "/src/visualizer/resources/assets/JetBrainsMono-Regular.ttf";
        io.Fonts->AddFontFromFileTTF(font_path.c_str(), 14.0f);

        applyDefaultStyle();

        // Configure components
        setScriptExecutor([this](const std::string& cmd) {
            return widgets::executeConsoleCommand(cmd, viewer_, event_bus_);
        });

        setFileSelectedCallback([this](const std::filesystem::path& path, bool is_dataset) {
            event_bus_->publish(LoadFileCommand{path, is_dataset});
            window_states_["file_browser"] = false;
        });

        scene_panel_->setOnDatasetLoad([this](const std::filesystem::path& path) {
            if (path.empty()) {
                window_states_["file_browser"] = true;
            } else {
                event_bus_->publish(LoadFileCommand{path, true});
            }
        });
    }

    void GuiManager::shutdown() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    void GuiManager::render() {
        // Start frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create context for this frame
        UIContext ctx{
            .viewer = viewer_,
            .event_bus = event_bus_,
            .console = console_.get(),
            .file_browser = file_browser_.get(),
            .window_states = &window_states_};

        // Render UI
        if (show_main_panel_) {
            panels::DrawMainPanel(ctx);
        }

        // Render windows
        if (window_states_["console"]) {
            console_->render(&window_states_["console"]);
        }

        if (window_states_["file_browser"]) {
            file_browser_->render(&window_states_["file_browser"]);
        }

        if (window_states_["camera_controls"]) {
            gui::windows::DrawCameraControls(&window_states_["camera_controls"]);
        }

        if (window_states_["scene_panel"]) {
            scene_panel_->render(&window_states_["scene_panel"]);
        }

        // End frame
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    void GuiManager::setupEventHandlers() {
        // Handle window visibility - only if ShowWindowEvent exists
        if (event_bus_) {
            // Handle log messages
            event_handler_ids_.push_back(
                event_bus_->subscribe<LogMessageEvent>([this](const LogMessageEvent& e) {
                    const char* level = "";
                    switch (e.level) {
                    case LogMessageEvent::Level::Info: level = "Info"; break;
                    case LogMessageEvent::Level::Warning: level = "Warning"; break;
                    case LogMessageEvent::Level::Error: level = "Error"; break;
                    case LogMessageEvent::Level::Debug: level = "Debug"; break;
                    }

                    if (e.source) {
                        console_->addLog("%s [%s]: %s", level, e.source->c_str(), e.message.c_str());
                    } else {
                        console_->addLog("%s: %s", level, e.message.c_str());
                    }
                }));
        }
    }

    void GuiManager::applyDefaultStyle() {
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
        style.WindowPadding = ImVec2(6.0f, 6.0f);
        style.WindowRounding = 6.0f;
        style.WindowBorderSize = 0.0f;
        style.FrameRounding = 2.0f;

        ImGui::StyleColorsLight();
    }

    void GuiManager::showWindow(const std::string& name, bool show) {
        window_states_[name] = show;
    }

    void GuiManager::toggleWindow(const std::string& name) {
        window_states_[name] = !window_states_[name];
    }

    void GuiManager::addConsoleLog(const char* fmt, ...) {
        if (console_) {
            va_list args;
            va_start(args, fmt);
            char buf[1024];
            vsnprintf(buf, sizeof(buf), fmt, args);
            va_end(args);
            console_->addLog("%s", buf);
        }
    }

    bool GuiManager::showCropBox() const {
        return panels::getCropBoxState().show_crop_box;
    }

    bool GuiManager::useCropBox() const {
        return panels::getCropBoxState().use_crop_box;
    }

    void GuiManager::setScriptExecutor(std::function<std::string(const std::string&)> executor) {
        if (console_) {
            console_->setExecutor(executor);
        }
    }

    void GuiManager::setFileSelectedCallback(std::function<void(const std::filesystem::path&, bool)> callback) {
        if (file_browser_) {
            file_browser_->setOnFileSelected(callback);
        }
    }
} // namespace gs::gui