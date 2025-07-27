#include "visualizer/gui_manager.hpp"
#include "config.h"
#include "visualizer/detail.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdarg>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <print>

namespace gs {
    namespace gui {

        // ============================================================================
        // ScriptingConsole Implementation
        // ============================================================================

        ScriptingConsole::ScriptingConsole() {
            clearLog();
            // Set default callback that just echoes input
            execute_callback_ = [](const std::string& input) -> std::string {
                return "Echo: " + input;
            };
        }

        void ScriptingConsole::clearLog() {
            output_buffer_.clear();
        }

        void ScriptingConsole::addLog(const char* fmt, ...) {
            char buf[1024];
            va_list args;
            va_start(args, fmt);
            vsnprintf(buf, sizeof(buf), fmt, args);
            buf[sizeof(buf) - 1] = 0;
            va_end(args);

            output_buffer_.push_back(std::string(buf));

            while (output_buffer_.size() > max_output_lines_) {
                output_buffer_.erase(output_buffer_.begin());
            }

            scroll_to_bottom_ = true;
        }

        void ScriptingConsole::executeCommand(const std::string& command) {
            addLog(">>> %s", command.c_str());

            // Add to history
            history_.push_back(command);

            // Execute command through callback
            if (execute_callback_) {
                try {
                    std::string result = execute_callback_(command);
                    if (!result.empty()) {
                        addLog("%s", result.c_str());
                    }
                } catch (const std::exception& e) {
                    addLog("Error: %s", e.what());
                }
            }

            scroll_to_bottom_ = true;
        }

        int ScriptingConsole::textEditCallbackStub(ImGuiInputTextCallbackData* data) {
            ScriptingConsole* console = (ScriptingConsole*)data->UserData;
            return console->textEditCallback(data);
        }

        int ScriptingConsole::textEditCallback(ImGuiInputTextCallbackData* data) {
            switch (data->EventFlag) {
            case ImGuiInputTextFlags_CallbackCompletion:
                // Handle tab completion here if needed
                break;

            case ImGuiInputTextFlags_CallbackHistory: {
                const int prev_history_pos = history_pos_;
                if (data->EventKey == ImGuiKey_UpArrow) {
                    if (history_pos_ == -1)
                        history_pos_ = static_cast<int>(history_.size()) - 1;
                    else if (history_pos_ > 0)
                        history_pos_--;
                } else if (data->EventKey == ImGuiKey_DownArrow) {
                    if (history_pos_ != -1) {
                        if (++history_pos_ >= static_cast<int>(history_.size()))
                            history_pos_ = -1;
                    }
                }

                if (prev_history_pos != history_pos_) {
                    const char* history_str = (history_pos_ >= 0) ? history_[history_pos_].c_str() : "";
                    data->DeleteChars(0, data->BufTextLen);
                    data->InsertChars(0, history_str);
                }
            } break;
            }
            return 0;
        }

        void ScriptingConsole::render(bool* p_open) {
            ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.08f, 0.95f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.1f, 0.1f, 0.15f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.15f, 0.15f, 0.25f, 1.0f));

            if (!ImGui::Begin("Scripting Console", p_open, ImGuiWindowFlags_MenuBar)) {
                ImGui::End();
                ImGui::PopStyleColor(4);
                return;
            }

            // Menu bar
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("Console")) {
                    if (ImGui::MenuItem("Clear", "Ctrl+L")) {
                        clearLog();
                    }
                    if (ImGui::MenuItem("Copy Output")) {
                        std::string output;
                        for (const auto& line : output_buffer_) {
                            output += line + "\n";
                        }
                        ImGui::SetClipboardText(output.c_str());
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }

            // Output area
            const float footer_height_to_reserve = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
            if (ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footer_height_to_reserve), false, ImGuiWindowFlags_HorizontalScrollbar)) {
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));

                for (const auto& line : output_buffer_) {
                    ImVec4 color;
                    bool has_color = false;

                    // Color coding for different types of output
                    if (line.find(">>>") == 0) {
                        color = ImVec4(0.8f, 0.8f, 0.2f, 1.0f); // Yellow for commands
                        has_color = true;
                    } else if (line.find("Error:") == 0) {
                        color = ImVec4(1.0f, 0.3f, 0.3f, 1.0f); // Red for errors
                        has_color = true;
                    } else if (line.find("Info:") == 0 || line.find("GPU Memory") == 0 ||
                               line.find("Model Information") == 0 || line.find("Training Status") == 0) {
                        color = ImVec4(0.3f, 0.8f, 0.3f, 1.0f); // Green for info
                        has_color = true;
                    }

                    if (has_color)
                        ImGui::PushStyleColor(ImGuiCol_Text, color);

                    ImGui::TextUnformatted(line.c_str());

                    if (has_color)
                        ImGui::PopStyleColor();
                }

                ImGui::PopStyleVar();

                if (scroll_to_bottom_ || ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
                    ImGui::SetScrollHereY(1.0f);
                scroll_to_bottom_ = false;
            }
            ImGui::EndChild();

            // Command input
            ImGui::Separator();

            // Input field - fix colors for visibility
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.25f, 0.25f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));

            bool reclaim_focus = false;
            ImGuiInputTextFlags input_text_flags = ImGuiInputTextFlags_EnterReturnsTrue |
                                                   ImGuiInputTextFlags_CallbackCompletion |
                                                   ImGuiInputTextFlags_CallbackHistory;

            ImGui::PushItemWidth(-1);
            if (ImGui::InputText("##input", input_buffer_, sizeof(input_buffer_),
                                 input_text_flags, &textEditCallbackStub, (void*)this)) {

                std::string command = input_buffer_;
                if (!command.empty()) {
                    executeCommand(command);
                    input_buffer_[0] = 0;
                    reclaim_focus = true;
                }
            }
            ImGui::PopItemWidth();
            ImGui::PopStyleColor(4);

            // Auto-focus on window appearing
            ImGui::SetItemDefaultFocus();
            if (reclaim_focus)
                ImGui::SetKeyboardFocusHere(-1);

            ImGui::End();
            ImGui::PopStyleColor(4);
        }

        void ScriptingConsole::setExecutor(std::function<std::string(const std::string&)> executor) {
            execute_callback_ = executor;
        }

        // ============================================================================
        // FileBrowser Implementation
        // ============================================================================

        FileBrowser::FileBrowser() {
            current_path_ = std::filesystem::current_path().string();
        }

        void FileBrowser::render(bool* p_open) {
            ImGui::SetNextWindowSize(ImVec2(700, 450), ImGuiCond_FirstUseEver);

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.15f, 0.95f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));

            if (!ImGui::Begin("File Browser", p_open, ImGuiWindowFlags_MenuBar)) {
                ImGui::End();
                ImGui::PopStyleColor(2);
                return;
            }

            // Menu bar
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("Quick Access")) {
                    if (ImGui::MenuItem("Current Directory")) {
                        current_path_ = std::filesystem::current_path().string();
                    }
                    if (ImGui::MenuItem("Home")) {
                        current_path_ = std::filesystem::path(std::getenv("HOME") ? std::getenv("HOME") : "/").string();
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }

            // Current path display
            ImGui::Text("Current Path: %s", current_path_.c_str());
            ImGui::Separator();

            // File list
            if (ImGui::BeginChild("FileList", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 2), true)) {
                std::filesystem::path current_path(current_path_);

                // Parent directory
                if (current_path.has_parent_path()) {
                    if (ImGui::Selectable("../", false, ImGuiSelectableFlags_DontClosePopups)) {
                        current_path_ = current_path.parent_path().string();
                        selected_file_.clear();
                    }
                }

                // List directories first
                std::vector<std::filesystem::directory_entry> dirs;
                std::vector<std::filesystem::directory_entry> files;

                try {
                    for (const auto& entry : std::filesystem::directory_iterator(current_path)) {
                        if (entry.is_directory()) {
                            dirs.push_back(entry);
                        } else if (entry.is_regular_file()) {
                            auto ext = entry.path().extension().string();
                            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                            // Show only relevant files
                            if (ext == ".ply" || ext == ".json" ||
                                entry.path().filename() == "cameras.bin" ||
                                entry.path().filename() == "transforms.json" ||
                                entry.path().filename() == "transforms_train.json") {
                                files.push_back(entry);
                            }
                        }
                    }
                } catch (const std::filesystem::filesystem_error& e) {
                    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", e.what());
                }

                // Sort entries
                std::sort(dirs.begin(), dirs.end(), [](const auto& a, const auto& b) {
                    return a.path().filename() < b.path().filename();
                });
                std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
                    return a.path().filename() < b.path().filename();
                });

                // Display directories
                for (const auto& dir : dirs) {
                    std::string dirname = "[DIR] " + dir.path().filename().string();
                    bool is_selected = (selected_file_ == dir.path().string());

                    // Check if this is a dataset directory
                    bool is_dataset = false;
                    if (std::filesystem::exists(dir.path() / "sparse" / "0" / "cameras.bin") ||
                        std::filesystem::exists(dir.path() / "sparse" / "cameras.bin") ||
                        std::filesystem::exists(dir.path() / "transforms.json") ||
                        std::filesystem::exists(dir.path() / "transforms_train.json")) {
                        is_dataset = true;
                    }

                    // Color code dataset directories
                    if (is_dataset) {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.5f, 0.9f, 1.0f));
                        dirname += " [Dataset]";
                    }

                    if (ImGui::Selectable(dirname.c_str(), is_selected,
                                          ImGuiSelectableFlags_AllowDoubleClick | ImGuiSelectableFlags_DontClosePopups)) {
                        if (ImGui::IsMouseDoubleClicked(0)) {
                            // Double-click enters the directory
                            current_path_ = dir.path().string();
                            selected_file_.clear();
                        } else {
                            // Single click selects the directory
                            selected_file_ = dir.path().string();
                        }
                    }

                    if (is_dataset) {
                        ImGui::PopStyleColor();
                    }
                }

                // Display files
                for (const auto& file : files) {
                    std::string filename = file.path().filename().string();
                    bool is_selected = (selected_file_ == file.path().string());

                    // Color code by type
                    ImVec4 color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
                    if (file.path().extension() == ".ply") {
                        color = ImVec4(0.3f, 0.8f, 0.3f, 1.0f); // Green for PLY
                    } else if (filename == "cameras.bin" || filename == "transforms.json" ||
                               filename == "transforms_train.json") {
                        color = ImVec4(0.3f, 0.5f, 0.9f, 1.0f); // Blue for dataset files
                    }

                    ImGui::PushStyleColor(ImGuiCol_Text, color);
                    if (ImGui::Selectable(filename.c_str(), is_selected, ImGuiSelectableFlags_DontClosePopups)) {
                        selected_file_ = file.path().string();
                    }
                    ImGui::PopStyleColor();
                }
            }
            ImGui::EndChild();

            // Selected file display
            if (!selected_file_.empty()) {
                ImGui::Text("Selected: %s", std::filesystem::path(selected_file_).filename().string().c_str());
            } else {
                ImGui::TextDisabled("No file selected");
            }

            // Action buttons
            ImGui::Separator();

            bool can_load = !selected_file_.empty();

            if (!can_load) {
                ImGui::BeginDisabled();
            }

            // Detect file type and show appropriate button
            if (can_load) {
                std::filesystem::path selected_path(selected_file_);

                // Check if it's a directory
                if (std::filesystem::is_directory(selected_path)) {
                    // Check for dataset types
                    bool is_colmap_dataset = false;
                    bool is_transforms_dataset = false;

                    if (std::filesystem::exists(selected_path / "sparse" / "0" / "cameras.bin") ||
                        std::filesystem::exists(selected_path / "sparse" / "cameras.bin")) {
                        is_colmap_dataset = true;
                    }

                    if (std::filesystem::exists(selected_path / "transforms.json") ||
                        std::filesystem::exists(selected_path / "transforms_train.json")) {
                        is_transforms_dataset = true;
                    }

                    if (is_colmap_dataset || is_transforms_dataset) {
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.8f, 1.0f));
                        if (ImGui::Button("Load Dataset", ImVec2(120, 0))) {
                            if (on_file_selected_) {
                                on_file_selected_(selected_path, true); // true = dataset
                                *p_open = false;
                            }
                        }
                        ImGui::PopStyleColor();

                        ImGui::SameLine();
                        ImGui::TextDisabled(is_colmap_dataset ? "(COLMAP)" : "(Transforms)");
                    } else {
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
                        if (ImGui::Button("Enter Directory", ImVec2(120, 0))) {
                            current_path_ = selected_path.string();
                            selected_file_.clear();
                        }
                        ImGui::PopStyleColor();

                        ImGui::SameLine();
                        ImGui::TextDisabled("(Not a dataset)");
                    }
                } else {
                    // It's a file
                    auto ext = selected_path.extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                    if (ext == ".ply") {
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                        if (ImGui::Button("Load PLY", ImVec2(120, 0))) {
                            if (on_file_selected_) {
                                on_file_selected_(selected_path, false); // false = PLY file
                                *p_open = false;
                            }
                        }
                        ImGui::PopStyleColor();
                    }
                }
            }

            if (!can_load) {
                ImGui::EndDisabled();
            }

            ImGui::SameLine();

            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                *p_open = false;
                selected_file_.clear();
            }

            ImGui::End();
            ImGui::PopStyleColor(2);
        }

        void FileBrowser::setOnFileSelected(std::function<void(const std::filesystem::path&, bool)> callback) {
            on_file_selected_ = callback;
        }

        void FileBrowser::setCurrentPath(const std::filesystem::path& path) {
            current_path_ = path.string();
        }

        // ============================================================================
        // CameraControlsWindow Implementation
        // ============================================================================

        void CameraControlsWindow::render(bool* p_open) {
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.15f, 0.9f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

            if (ImGui::Begin("Camera Controls", p_open, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::Text("Camera Controls:");
                ImGui::Separator();

                // Table for better formatting
                if (ImGui::BeginTable("camera_controls_table", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 400.0f);
                    ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableHeadersRow();

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Local Translate Camera");
                    ImGui::TableNextColumn();
                    ImGui::Text("Left Mouse + Drag");

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Local Rotate Camera (Pitch/Yaw)");
                    ImGui::TableNextColumn();
                    ImGui::Text("Right Mouse + Drag");

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Rotate Around Scene Center");
                    ImGui::TableNextColumn();
                    ImGui::Text("Middle Mouse + Drag");

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Zoom");
                    ImGui::TableNextColumn();
                    ImGui::Text("Mouse Scroll");

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Roll Camera");
                    ImGui::TableNextColumn();
                    ImGui::Text("R + Mouse Scroll");

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Move forward, backward, left and right within the scene");
                    ImGui::TableNextColumn();
                    ImGui::Text("w, s, a, d keys");

                    ImGui::EndTable();
                }

                ImGui::Separator();
            }
            ImGui::End();
            ImGui::PopStyleColor(4);
        }

        // ============================================================================
        // TrainingControlsPanel Implementation
        // ============================================================================

        void TrainingControlsPanel::render(Trainer* trainer, State& state, std::shared_ptr<ViewerNotifier> notifier) {
            if (!trainer)
                return;

            ImGui::Separator();
            ImGui::Text("Training Control");
            ImGui::Separator();

            bool is_training = trainer->is_running();
            bool is_paused = trainer->is_paused();
            bool is_complete = trainer->is_training_complete();
            bool has_stopped = trainer->has_stopped();

            // Show appropriate controls based on state
            if (!state.training_started && !is_training) {
                renderStartButton(state);
            } else if (is_complete || has_stopped) {
                // Training finished - show status
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                                   has_stopped ? "Training Stopped!" : "Training Complete!");
            } else {
                renderRunningControls(trainer, state);
            }

            // Show save progress feedback
            if (state.save_in_progress) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - state.save_start_time).count();
                if (elapsed < 2000) {
                    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Checkpoint saved!");
                } else {
                    state.save_in_progress = false;
                }
            }

            renderStatus(trainer, state);
        }

        void TrainingControlsPanel::renderStartButton(State& state) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            if (ImGui::Button("Start Training", ImVec2(-1, 0))) {
                state.manual_start_triggered = true;
                state.training_started = true;
            }
            ImGui::PopStyleColor(2);
        }

        void TrainingControlsPanel::renderRunningControls(Trainer* trainer, State& state) {
            bool is_paused = trainer->is_paused();

            if (is_paused) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
                if (ImGui::Button("Resume", ImVec2(-1, 0))) {
                    trainer->request_resume();
                }
                ImGui::PopStyleColor(2);

                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
                if (ImGui::Button("Stop Permanently", ImVec2(-1, 0))) {
                    trainer->request_stop();
                }
                ImGui::PopStyleColor(2);
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.5f, 0.1f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.6f, 0.2f, 1.0f));
                if (ImGui::Button("Pause", ImVec2(-1, 0))) {
                    trainer->request_pause();
                }
                ImGui::PopStyleColor(2);
            }

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.4f, 0.7f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
            if (ImGui::Button("Save Checkpoint", ImVec2(-1, 0))) {
                trainer->request_save();
                state.save_in_progress = true;
                state.save_start_time = std::chrono::steady_clock::now();
            }
            ImGui::PopStyleColor(2);
        }

        void TrainingControlsPanel::renderStatus(Trainer* trainer, State& state) {
            ImGui::Separator();
            int current_iter = trainer->get_current_iteration();
            float current_loss = trainer->get_current_loss();
            bool is_training = trainer->is_running();
            bool is_paused = trainer->is_paused();
            bool is_complete = trainer->is_training_complete();

            ImGui::Text("Status: %s", is_complete ? "Complete" : (is_paused ? "Paused" : (is_training ? "Training" : "Ready")));
            ImGui::Text("Iteration: %d", current_iter);
            ImGui::Text("Loss: %.6f", current_loss);
        }

        // ============================================================================
        // GuiManager Implementation
        // ============================================================================

        GuiManager::GuiManager(GSViewer* viewer) : viewer_(viewer) {
            scripting_console_ = std::make_unique<ScriptingConsole>();
            file_browser_ = std::make_unique<FileBrowser>();
            camera_controls_ = std::make_unique<CameraControlsWindow>();
            training_controls_ = std::make_unique<TrainingControlsPanel>();
            crop_box_panel_ = std::make_unique<CropBoxPanel>();
            crop_box_panel_->crop_box_ = viewer->crop_box_;
        }

        GuiManager::~GuiManager() = default;

        void GuiManager::init() {
            // Setup Dear ImGui context
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGuiIO& io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
            io.ConfigWindowsMoveFromTitleBarOnly = true;
            ImGui::StyleColorsLight();

            // Setup Platform/Renderer backends
            const char* glsl_version = "#version 430";
            auto window = viewer_->getWindow();
            ImGui_ImplGlfw_InitForOpenGL(window, true);
            ImGui_ImplOpenGL3_Init(glsl_version);

            // Set Fonts
            std::string font_path = std::string(PROJECT_ROOT_PATH) +
                                    "/include/visualizer/assets/JetBrainsMono-Regular.ttf";
            io.Fonts->AddFontFromFileTTF(font_path.c_str(), 14.0f);

            // Set Windows option
            window_flags_ |= ImGuiWindowFlags_NoScrollbar;
            window_flags_ |= ImGuiWindowFlags_NoResize;

            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 2.0f);

            ImGuiStyle& style = ImGui::GetStyle();
            style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
            style.WindowPadding = ImVec2(6.0f, 6.0f);
            style.WindowRounding = 6.0f;
            style.WindowBorderSize = 0.0f;
        }

        void GuiManager::shutdown() {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }

        void GuiManager::beginFrame() {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            any_window_active_ = ImGui::IsAnyItemActive();
        }

        void GuiManager::endFrame() {
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        void GuiManager::render() {
            beginFrame();

            if (show_main_panel_) {
                renderMainPanel();
            }

            if (show_file_browser_) {
                file_browser_->render(&show_file_browser_);
            }

            if (show_scripting_console_) {
                scripting_console_->render(&show_scripting_console_);
            }

            if (show_camera_controls_) {
                camera_controls_->render(&show_camera_controls_);
            }

            endFrame();
        }

        void GuiManager::renderMainPanel() {
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));
            ImGui::Begin("Rendering Setting", nullptr, window_flags_);
            ImGui::SetWindowSize(ImVec2(300, 0));

            // File Browser button - always visible
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.3f, 0.6f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.4f, 0.7f, 1.0f));
            if (ImGui::Button("Open File Browser", ImVec2(-1, 0))) {
                show_file_browser_ = true;
            }
            ImGui::PopStyleColor(2);

            ImGui::Separator();

            renderModeStatus();

            // Mode-specific controls
            if (viewer_->getCurrentMode() == GSViewer::ViewerMode::Training && viewer_->getTrainer()) {
                training_controls_->render(viewer_->getTrainer(), training_state_, viewer_->getNotifier());

                // Handle the start trigger
                if (training_state_.manual_start_triggered) {
                    viewer_->startTraining();
                    training_state_.manual_start_triggered = false;
                }
            } else if (viewer_->getCurrentMode() == GSViewer::ViewerMode::PLYViewer && viewer_->getStandaloneModel()) {
                // PLY viewer info
                ImGui::Separator();
                ImGui::Text("Model Information");
                ImGui::Separator();
                auto model = viewer_->getStandaloneModel();
                ImGui::Text("Gaussians: %lld", model->size());
                ImGui::Text("SH Degree: %d", model->get_active_sh_degree());
                ImGui::Text("Scene Scale: %.3f", model->get_scene_scale());

                // Disabled training button for PLY mode
                ImGui::Separator();
                ImGui::BeginDisabled(true);
                ImGui::Button("Start Training", ImVec2(-1, 0));
                ImGui::EndDisabled();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Training not available for PLY files");
            }

            // Only show these settings if we have data loaded
            if (viewer_->getCurrentMode() != GSViewer::ViewerMode::Empty) {
                renderRenderingSettings();
            }

            // Show training progress for training mode
            if (viewer_->getCurrentMode() == GSViewer::ViewerMode::Training && viewer_->getTrainer()) {
                renderProgressInfo();
            }

            // GPU usage - always show if we have data
            if (viewer_->getCurrentMode() != GSViewer::ViewerMode::Empty) {
                float gpuUsage = viewer_->getGPUUsage();
                char gpuText[64];
                std::snprintf(gpuText, sizeof(gpuText), "GPU Usage: %.1f%%", gpuUsage);
                ImGui::ProgressBar(gpuUsage / 100.0f, ImVec2(-1, 20), gpuText);
            }

            // Bottom buttons
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.4f, 0.4f, 0.7f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.5f, 0.5f, 0.8f, 1.0f));
            if (ImGui::Button("Show Camera Controls", ImVec2(-1, 0))) {
                show_camera_controls_ = true;
            }
            ImGui::PopStyleColor(2);

            if (show_crop_box_panel_) {
                ImGui::Separator();
                crop_box_panel_->render();
            }

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.3f, 1.0f));
            if (ImGui::Button("Open Console", ImVec2(-1, 0))) {
                show_scripting_console_ = true;
                scripting_console_->addLog("Console opened. Type 'help' for available commands.");
            }
            ImGui::PopStyleColor(2);

            ImGui::End();
            ImGui::PopStyleColor();
        }

        void GuiManager::renderModeStatus() {
            // Show current mode status
            switch (viewer_->getCurrentMode()) {
            case GSViewer::ViewerMode::Empty:
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No data loaded");
                ImGui::Text("Use File Browser to load:");
                ImGui::BulletText("PLY file for viewing");
                ImGui::BulletText("Dataset for training");
                break;

            case GSViewer::ViewerMode::PLYViewer:
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "PLY Viewer Mode");
                ImGui::Text("File: %s", viewer_->getCurrentPLYPath().filename().string().c_str());
                break;

            case GSViewer::ViewerMode::Training:
                ImGui::TextColored(ImVec4(0.2f, 0.5f, 0.8f, 1.0f), "Training Mode");
                ImGui::Text("Dataset: %s", viewer_->getCurrentDatasetPath().filename().string().c_str());
                break;
            }
        }

        void GuiManager::renderRenderingSettings() {
            auto config = viewer_->getRenderingConfig();

            ImGui::Separator();
            ImGui::Text("Rendering Settings");
            ImGui::Separator();

            ImGui::SetNextItemWidth(200);
            ImGui::SliderFloat("##scale_slider", &config->scaling_modifier, 0.01f, 3.0f, "Scale=%.2f");
            ImGui::SameLine();
            if (ImGui::Button("Reset##scale", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
                config->scaling_modifier = 1.0f;
            }

            ImGui::SetNextItemWidth(200);
            ImGui::SliderFloat("##fov_slider", &config->fov, 45.0f, 120.0f, "FoV=%.2f");
            ImGui::SameLine();
            if (ImGui::Button("Reset##fov", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
                config->fov = 75.0f;
            }

            // Show render mode
#ifdef CUDA_GL_INTEROP_ENABLED
            ImGui::Text("Render Mode: GPU Direct (Interop)");
#else
            ImGui::Text("Render Mode: CPU Copy");
#endif
        }

        void GuiManager::renderProgressInfo() {
            auto info = viewer_->getTrainingInfo();
            int current_iter = info->curr_iterations_.load();
            int total_iter = info->total_iterations_.load();
            int num_splats = info->num_splats_.load();

            std::vector<float> loss_data;
            {
                std::lock_guard<std::mutex> lock(info->loss_buffer_mutex_);
                loss_data.assign(info->loss_buffer_.begin(), info->loss_buffer_.end());
            }

            float fraction = total_iter > 0 ? float(current_iter) / float(total_iter) : 0.0f;
            char overlay_text[64];
            std::snprintf(overlay_text, sizeof(overlay_text), "%d / %d", current_iter, total_iter);
            ImGui::ProgressBar(fraction, ImVec2(-1, 20), overlay_text);

            if (loss_data.size() > 0) {
                auto [min_it, max_it] = std::minmax_element(loss_data.begin(), loss_data.end());
                float min_val = *min_it, max_val = *max_it;

                if (min_val == max_val) {
                    min_val -= 1.0f;
                    max_val += 1.0f;
                } else {
                    float margin = (max_val - min_val) * 0.05f;
                    min_val -= margin;
                    max_val += margin;
                }

                char loss_label[64];
                std::snprintf(loss_label, sizeof(loss_label), "Loss: %.4f", loss_data.back());

                ImGui::PlotLines(
                    "##Loss",
                    loss_data.data(),
                    static_cast<int>(loss_data.size()),
                    0,
                    loss_label,
                    min_val,
                    max_val,
                    ImVec2(-1, 50));
            }

            ImGui::Text("num Splats: %d", num_splats);
        }

        void GuiManager::showFileBrowser(bool show) {
            show_file_browser_ = show;
        }

        void GuiManager::showScriptingConsole(bool show) {
            show_scripting_console_ = show;
        }

        void GuiManager::showCameraControls(bool show) {
            show_camera_controls_ = show;
        }

        void GuiManager::setScriptExecutor(std::function<std::string(const std::string&)> executor) {
            scripting_console_->setExecutor(executor);
        }

        void GuiManager::setFileSelectedCallback(std::function<void(const std::filesystem::path&, bool)> callback) {
            file_browser_->setOnFileSelected(callback);
        }

        void GuiManager::addConsoleLog(const char* fmt, ...) {
            if (scripting_console_) {
                va_list args;
                va_start(args, fmt);
                char buf[1024];
                vsnprintf(buf, sizeof(buf), fmt, args);
                va_end(args);
                scripting_console_->addLog("%s", buf);
            }
        }

        bool GuiManager::showCropBox() const {
            if (crop_box_panel_)
                return crop_box_panel_->show_crop_box_;
            else
                return false;
        }

        bool GuiManager::useCropBox() const {
            if (crop_box_panel_)
                return crop_box_panel_->use_crop_box_;
            else
                return false;
        }

        void CropBoxPanel::render() {
            if (ImGui::CollapsingHeader("Clip Box")) {
                ImGui::Checkbox("Show Clip Box", &show_crop_box_);
                ImGui::Checkbox("Use Clip Box", &use_crop_box_);

                if (show_crop_box_ && crop_box_) {
                    if (!crop_box_->isInitialized()) {
                        return; // the manager must init crop box
                    }

                    // Color picker
                    static float bbox_color[3] = {1.0f, 1.0f, 0.0f}; // Yellow default
                    if (ImGui::ColorEdit3("Box Color", bbox_color)) {
                        crop_box_->setColor(glm::vec3(bbox_color[0], bbox_color[1], bbox_color[2]));
                    }

                    // Line width and Reset button aligned
                    static float line_width = 2.0f;
                    float available_width = ImGui::GetContentRegionAvail().x;
                    float button_width = 120.0f; // Adjust this value as needed
                    float slider_width = available_width - button_width - ImGui::GetStyle().ItemSpacing.x;

                    ImGui::SetNextItemWidth(slider_width);
                    if (ImGui::SliderFloat("Line Width", &line_width, 0.5f, 10.0f)) {
                        crop_box_->setLineWidth(line_width);
                    }

                    // Reset button on next line
                    if (ImGui::Button("Reset to Default")) {
                        crop_box_->setBounds(glm::vec3(-1.0f), glm::vec3(1.0f));
                    }

                    // Manual bounds adjustment
                    if (ImGui::TreeNode("Manual Bounds")) {
                        glm::vec3 current_min = crop_box_->getMinBounds();
                        glm::vec3 current_max = crop_box_->getMaxBounds();

                        float min_bounds[3] = {current_min.x, current_min.y, current_min.z};
                        float max_bounds[3] = {current_max.x, current_max.y, current_max.z};

                        bool bounds_changed = false;

                        // Define reasonable ranges for the sliders (adjust these values as needed)
                        const float min_range = -8.0f;
                        const float max_range = 8.0f;

                        // Min Bounds - each slider on separate line
                        ImGui::Text("Min Bounds:");
                        float min_bounds_x = min_bounds[0];
                        bounds_changed |= ImGui::SliderFloat("Min X", &min_bounds_x, min_range, max_range, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                        min_bounds_x = std::min(min_bounds_x, max_bounds[0]);
                        min_bounds[0] = min_bounds_x;

                        float min_bounds_y = min_bounds[1];
                        bounds_changed |= ImGui::SliderFloat("Min Y", &min_bounds_y, min_range, max_range, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                        min_bounds_y = std::min(min_bounds_y, max_bounds[1]);
                        min_bounds[1] = min_bounds_y;

                        float min_bounds_z = min_bounds[2];
                        bounds_changed |= ImGui::SliderFloat("Min Z", &min_bounds_z, min_range, max_range, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                        min_bounds_z = std::min(min_bounds_z, max_bounds[2]);
                        min_bounds[2] = min_bounds_z;

                        // Max Bounds - each slider on separate line
                        ImGui::Text("Max Bounds:");
                        float max_bounds_x = max_bounds[0];
                        bounds_changed |= ImGui::SliderFloat("Max X", &max_bounds_x, min_range, max_range, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                        max_bounds_x = std::max(max_bounds_x, min_bounds[0]);
                        max_bounds[0] = max_bounds_x;

                        float max_bounds_y = max_bounds[1];
                        bounds_changed |= ImGui::SliderFloat("Max Y", &max_bounds_y, min_range, max_range, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                        max_bounds_y = std::max(max_bounds_y, min_bounds[1]);
                        max_bounds[1] = max_bounds_y;

                        float max_bounds_z = max_bounds[2];
                        bounds_changed |= ImGui::SliderFloat("Max Z", &max_bounds_z, min_range, max_range, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                        max_bounds_z = std::max(max_bounds_z, min_bounds[2]);
                        max_bounds[2] = max_bounds_z;

                        if (bounds_changed) {
                            crop_box_->setBounds(
                                glm::vec3(min_bounds[0], min_bounds[1], min_bounds[2]),
                                glm::vec3(max_bounds[0], max_bounds[1], max_bounds[2]));
                        }

                        // Display current info
                        glm::vec3 center = crop_box_->getCenter();
                        glm::vec3 size = crop_box_->getSize();

                        ImGui::Text("Center: (%.3f, %.3f, %.3f)", center.x, center.y, center.z);
                        ImGui::Text("Size: (%.3f, %.3f, %.3f)", size.x, size.y, size.z);

                        ImGui::TreePop();
                    }
                }
            }
        }

    } // namespace gui
} // namespace gs