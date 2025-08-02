#include "gui/gui_manager.hpp"
#include "gui/panels/main_panel.hpp"
#include "gui/panels/scene_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/windows/camera_controls.hpp"
#include "gui/windows/file_browser.hpp"
#include "gui/windows/scripting_console.hpp"
#include "tools/crop_box_tool.hpp"
#include "visualizer_impl.hpp"

#include <GLFW/glfw3.h>
#include <chrono>
#include <cstdarg>
#include <format>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_internal.h>

namespace gs::gui {

    GuiManager::GuiManager(visualizer::VisualizerImpl* viewer)
        : viewer_(viewer) {

        // Create components
        console_ = std::make_unique<ScriptingConsole>();
        file_browser_ = std::make_unique<FileBrowser>();
        scene_panel_ = std::make_unique<ScenePanel>();

        // Initialize window states
        window_states_["console"] = false;
        window_states_["file_browser"] = false;
        window_states_["camera_controls"] = false;
        window_states_["scene_panel"] = true;

        // Initialize speed overlay state
        speed_overlay_visible_ = false;
        speed_overlay_duration_ = std::chrono::milliseconds(3000); // 3 seconds

        setupEventHandlers();
    }

    GuiManager::~GuiManager() {
        // Cleanup handled automatically
    }

    void GuiManager::init() {
        // ImGui initialization
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;   // Enable Docking
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport
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
            return widgets::executeConsoleCommand(cmd, viewer_);
        });

        setFileSelectedCallback([this](const std::filesystem::path& path, bool is_dataset) {
            events::cmd::LoadFile{.path = path, .is_dataset = is_dataset}.emit();
            window_states_["file_browser"] = false;
        });

        scene_panel_->setOnDatasetLoad([this](const std::filesystem::path& path) {
            if (path.empty()) {
                window_states_["file_browser"] = true;
            } else {
                events::cmd::LoadFile{.path = path, .is_dataset = true}.emit();
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

        // Get viewport
        const ImGuiViewport* viewport = ImGui::GetMainViewport();

        // Define layout dimensions
        const float left_panel_width = 250.0f;  // Slimmer Rendering Settings panel
        const float right_panel_width = 200.0f; // Even slimmer Scene panel

        // Create context for this frame
        UIContext ctx{
            .viewer = viewer_,
            .console = console_.get(),
            .file_browser = file_browser_.get(),
            .window_states = &window_states_};

        // Draw the main panel (Rendering Settings) with proper positioning
        if (show_main_panel_) {
            ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(left_panel_width, viewport->WorkSize.y), ImGuiCond_Always);
            panels::DrawMainPanel(ctx);
        }

        // Draw Scene panel with proper positioning
        if (window_states_["scene_panel"]) {
            ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - right_panel_width, viewport->WorkPos.y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(right_panel_width, viewport->WorkSize.y), ImGuiCond_Always);
            scene_panel_->render(&window_states_["scene_panel"]);
        }

        // Render floating windows (these should remain movable)
        if (window_states_["console"]) {
            console_->render(&window_states_["console"]);
        }

        if (window_states_["file_browser"]) {
            file_browser_->render(&window_states_["file_browser"]);
        }

        if (window_states_["camera_controls"]) {
            gui::windows::DrawCameraControls(&window_states_["camera_controls"]);
        }

        // Render speed overlay if visible
        renderSpeedOverlay();

        // End frame
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows (for multi-viewport)
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }
    }

    void GuiManager::renderSpeedOverlay() {
        // Check if overlay should be hidden
        if (speed_overlay_visible_) {
            auto now = std::chrono::steady_clock::now();
            if (now - speed_overlay_start_time_ >= speed_overlay_duration_) {
                speed_overlay_visible_ = false;
                return;
            }
        }

        // Get viewport for positioning
        const ImGuiViewport* viewport = ImGui::GetMainViewport();

        // Position overlay in the center-top of the viewport
        const float overlay_width = 300.0f;
        const float overlay_height = 80.0f;
        const float padding = 20.0f;

        ImVec2 overlay_pos(
            viewport->WorkPos.x + (viewport->WorkSize.x - overlay_width) * 0.5f,
            viewport->WorkPos.y + padding);

        // Create overlay window
        ImGui::SetNextWindowPos(overlay_pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(overlay_width, overlay_height), ImGuiCond_Always);

        // Window flags to make it non-interactive and styled nicely
        ImGuiWindowFlags overlay_flags =
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoInputs |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoBringToFrontOnFocus;

        // Apply semi-transparent background
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 1.0f, 1.0f, 0.3f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);

        if (ImGui::Begin("##SpeedOverlay", nullptr, overlay_flags)) {
            // Calculate fade effect based on remaining time
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - speed_overlay_start_time_);
            auto remaining = speed_overlay_duration_ - elapsed;

            float fade_alpha = 1.0f;
            if (remaining < std::chrono::milliseconds(500)) {
                // Fade out in the last 500ms
                fade_alpha = static_cast<float>(remaining.count()) / 500.0f;
            }

            // Center the text
            ImVec2 window_size = ImGui::GetWindowSize();

            // Speed text
            std::string speed_text = std::format("WASD Speed: {:.2f}", current_speed_);
            ImVec2 speed_text_size = ImGui::CalcTextSize(speed_text.c_str());
            ImGui::SetCursorPos(ImVec2(
                (window_size.x - speed_text_size.x) * 0.5f,
                window_size.y * 0.3f));

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, fade_alpha));
            ImGui::Text("%s", speed_text.c_str());
            ImGui::PopStyleColor();

            // Max speed text
            std::string max_text = std::format("Max: {:.3f}", max_speed_);
            ImVec2 max_text_size = ImGui::CalcTextSize(max_text.c_str());
            ImGui::SetCursorPos(ImVec2(
                (window_size.x - max_text_size.x) * 0.5f,
                window_size.y * 0.6f));

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.8f, fade_alpha * 0.8f));
            ImGui::Text("%s", max_text.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::End();

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);
    }

    void GuiManager::showSpeedOverlay(float current_speed, float max_speed) {
        current_speed_ = current_speed;
        max_speed_ = max_speed;
        speed_overlay_visible_ = true;
        speed_overlay_start_time_ = std::chrono::steady_clock::now();
    }

    void GuiManager::setupEventHandlers() {
        using namespace events;

        // Handle window visibility
        cmd::ShowWindow::when([this](const auto& e) {
            showWindow(e.window_name, e.show);
        });

        // Handle speed change events (you'll need to add this event type)
        ui::SpeedChanged::when([this](const auto& e) {
            showSpeedOverlay(e.current_speed, e.max_speed);
        });

        // Handle log messages
        notify::Log::when([this](const auto& e) {
            const char* level = "";
            switch (e.level) {
            case notify::Log::Level::Info: level = "Info"; break;
            case notify::Log::Level::Warning: level = "Warning"; break;
            case notify::Log::Level::Error: level = "Error"; break;
            case notify::Log::Level::Debug: level = "Debug"; break;
            }

            if (!e.source.empty()) {
                console_->addLog("%s [%s]: %s", level, e.source.c_str(), e.message.c_str());
            } else {
                console_->addLog("%s: %s", level, e.message.c_str());
            }
        });

        // Handle console results
        ui::ConsoleResult::when([this](const auto& e) {
            console_->addLog("> %s", e.command.c_str());
            if (!e.result.empty()) {
                console_->addLog("%s", e.result.c_str());
            }
        });

        // Handle errors
        notify::Error::when([this](const auto& e) {
            addConsoleLog("ERROR: %s", e.message.c_str());
            if (!e.details.empty()) {
                addConsoleLog("Details: %s", e.details.c_str());
            }
        });

        // Handle success messages
        notify::Success::when([this](const auto& e) {
            addConsoleLog("SUCCESS: %s", e.message.c_str());
        });

        // Handle warnings
        notify::Warning::when([this](const auto& e) {
            addConsoleLog("WARNING: %s", e.message.c_str());
        });
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

    bool GuiManager::wantsInput() const {
        ImGuiIO& io = ImGui::GetIO();
        return io.WantCaptureMouse || io.WantCaptureKeyboard;
    }

    bool GuiManager::isAnyWindowActive() const {
        return ImGui::IsAnyItemActive() ||
               ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) ||
               ImGui::GetIO().WantCaptureMouse ||
               ImGui::GetIO().WantCaptureKeyboard;
    }

    bool GuiManager::showCropBox() const {
        if (auto* tool_manager = viewer_->getToolManager()) {
            if (auto* crop_tool = dynamic_cast<visualizer::CropBoxTool*>(
                    tool_manager->getTool("Crop Box"))) {
                return crop_tool->shouldShowBox();
            }
        }
        return false;
    }

    bool GuiManager::useCropBox() const {
        if (auto* tool_manager = viewer_->getToolManager()) {
            if (auto* crop_tool = dynamic_cast<visualizer::CropBoxTool*>(
                    tool_manager->getTool("Crop Box"))) {
                return crop_tool->shouldUseBox();
            }
        }
        return false;
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