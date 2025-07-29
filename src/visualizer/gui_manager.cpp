#include "visualizer/gui_manager.hpp"
#include "config.h"
#include "visualizer/detail.hpp"
#include "visualizer/event_response_handler.hpp"
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

        void FileBrowser::setSelectedPath(const std::filesystem::path& path) {
            selected_file_ = path.string();
            if (std::filesystem::is_directory(path)) {
                current_path_ = path.string();
            } else {
                current_path_ = path.parent_path().string();
            }
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

            // Get trainer manager from viewer
            auto gui_manager = static_cast<GuiManager*>(ImGui::GetIO().UserData);
            if (!gui_manager || !gui_manager->viewer_)
                return;

            auto viewer = gui_manager->viewer_;
            if (!viewer || !viewer->getTrainerManager())
                return;

            auto trainer_manager = viewer->getTrainerManager();
            auto training_state = trainer_manager->getState();

            // Show appropriate controls based on state
            switch (training_state) {
            case TrainerManager::State::Idle:
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No trainer loaded");
                break;

            case TrainerManager::State::Ready:
                renderStartButton(state);
                break;

            case TrainerManager::State::Running:
            case TrainerManager::State::Paused:
                renderRunningControls(trainer, state);
                break;

            case TrainerManager::State::Stopping:
                ImGui::TextColored(ImVec4(0.7f, 0.5f, 0.1f, 1.0f), "Stopping training...");
                break;

            case TrainerManager::State::Completed:
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Training Complete!");
                break;

            case TrainerManager::State::Error:
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Training Error!");
                if (!trainer_manager->getLastError().empty()) {
                    ImGui::TextWrapped("%s", trainer_manager->getLastError().c_str());
                }
                break;
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
                // Use event bus from parent GuiManager
                auto gui_manager = static_cast<GuiManager*>(ImGui::GetIO().UserData);
                if (gui_manager) {
                    gui_manager->publish(StartTrainingCommand{});
                }
                state.training_started = true;
            }
            ImGui::PopStyleColor(2);
        }

        void TrainingControlsPanel::renderRunningControls(Trainer* trainer, State& state) {
            auto gui_manager = static_cast<GuiManager*>(ImGui::GetIO().UserData);
            if (!gui_manager || !gui_manager->viewer_)
                return;

            auto viewer = gui_manager->viewer_;
            if (!viewer || !viewer->getTrainerManager())
                return;

            auto trainer_manager = viewer->getTrainerManager();
            bool is_paused = trainer_manager->isPaused();

            if (is_paused) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
                if (ImGui::Button("Resume", ImVec2(-1, 0))) {
                    gui_manager->publish(ResumeTrainingCommand{});
                }
                ImGui::PopStyleColor(2);

                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
                if (ImGui::Button("Stop Permanently", ImVec2(-1, 0))) {
                    gui_manager->publish(StopTrainingCommand{});
                }
                ImGui::PopStyleColor(2);
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.5f, 0.1f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.6f, 0.2f, 1.0f));
                if (ImGui::Button("Pause", ImVec2(-1, 0))) {
                    gui_manager->publish(PauseTrainingCommand{});
                }
                ImGui::PopStyleColor(2);
            }

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.4f, 0.7f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
            if (ImGui::Button("Save Checkpoint", ImVec2(-1, 0))) {
                gui_manager->publish(SaveCheckpointCommand{});
                state.save_in_progress = true;
                state.save_start_time = std::chrono::steady_clock::now();
            }
            ImGui::PopStyleColor(2);
        }

        void TrainingControlsPanel::renderStatus(Trainer* trainer, State& state) {
            ImGui::Separator();

            auto gui_manager = static_cast<GuiManager*>(ImGui::GetIO().UserData);
            if (!gui_manager)
                return;

            auto event_bus = gui_manager->getEventBus();
            if (!event_bus)
                return;

            // Query trainer state via events
            EventResponseHandler<QueryTrainerStateRequest, QueryTrainerStateResponse> handler(event_bus);
            auto response = handler.querySync(QueryTrainerStateRequest{});

            if (response) {
                const char* status_text = "Unknown";
                switch (response->state) {
                case QueryTrainerStateResponse::State::Idle: status_text = "Idle"; break;
                case QueryTrainerStateResponse::State::Ready: status_text = "Ready"; break;
                case QueryTrainerStateResponse::State::Running: status_text = "Training"; break;
                case QueryTrainerStateResponse::State::Paused: status_text = "Paused"; break;
                case QueryTrainerStateResponse::State::Stopping: status_text = "Stopping"; break;
                case QueryTrainerStateResponse::State::Completed: status_text = "Complete"; break;
                case QueryTrainerStateResponse::State::Error: status_text = "Error"; break;
                }

                ImGui::Text("Status: %s", status_text);
                ImGui::Text("Iteration: %d", response->current_iteration);
                ImGui::Text("Loss: %.6f", response->current_loss);

                if (response->error_message) {
                    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", response->error_message->c_str());
                }
            }
        }

        // ============================================================================
        // CropBoxPanel Implementation
        // ============================================================================

        // Updated CropBoxPanel implementation with split functions

        void CropBoxPanel::render() {
            if (ImGui::CollapsingHeader("Crop Box")) {
                ImGui::Checkbox("Show Crop Box", &show_crop_box_);
                ImGui::Checkbox("Use Crop Box", &use_crop_box_);

                if (show_crop_box_ && crop_box_) {
                    if (!crop_box_->isInitialized()) {
                        return; // the manager must init crop box
                    }

                    renderAppearanceControls();
                    renderRotationControls();
                    renderBoundsControls();
                }
            }
        }

        void CropBoxPanel::renderAppearanceControls() {
            // Color picker
            static float bbox_color[3] = {1.0f, 1.0f, 0.0f}; // Yellow default
            if (ImGui::ColorEdit3("Box Color", bbox_color)) {
                crop_box_->setColor(glm::vec3(bbox_color[0], bbox_color[1], bbox_color[2]));
            }

            // Line width
            static float line_width = 2.0f;
            float available_width = ImGui::GetContentRegionAvail().x;
            float button_width = 120.0f;
            float slider_width = available_width - button_width - ImGui::GetStyle().ItemSpacing.x;

            ImGui::SetNextItemWidth(slider_width);
            if (ImGui::SliderFloat("Line Width", &line_width, 0.5f, 10.0f)) {
                crop_box_->setLineWidth(line_width);
            }

            // Reset button
            if (ImGui::Button("Reset to Default")) {
                crop_box_->setBounds(glm::vec3(-1.0f), glm::vec3(1.0f));
                // Reset rotation angles to 0
                crop_box_->setworld2BBox(glm::mat4(1.0));
            }
        }

        void CropBoxPanel::renderRotationControls() {
            if (ImGui::TreeNode("Rotation")) {
                ImGui::Text("Ctrl+click for faster steps");
                ImGui::Text("Rotation around crop box axes:");

                renderRotationInputs();
                ImGui::TreePop();
            }
        }

        void CropBoxPanel::renderRotationInputs() {
            const float rotation_text_width = 110.0f;
            const float rotation_step = 1.0f;
            const float rotation_step_fast = 15.0f;

            bool rotation_changed = false;

            float diff_x = 0, diff_y = 0, diff_z = 0;

            float step = ImGui::GetIO().KeyCtrl ? rotation_step_fast : rotation_step;
            float repeat_rate = 0.05f;

            // Separate timers per axis
            static float rotate_timer_x = 0.0f;
            static float rotate_timer_y = 0.0f;
            static float rotate_timer_z = 0.0f;

            // --- X-axis ---
            ImGui::Text("X-axis:");
            ImGui::SameLine();
            ImGui::Text("RotX");

            if (ImGui::ArrowButton("##RotX_Up", ImGuiDir_Up)) {
                diff_x = step;
                rotation_changed = true;
                rotate_timer_x = 0.0f;
            }
            if (ImGui::IsItemActive()) {
                rotate_timer_x += ImGui::GetIO().DeltaTime;
                if (rotate_timer_x >= repeat_rate) {
                    diff_x = step;
                    rotation_changed = true;
                    rotate_timer_x = 0.0f;
                }
            }

            ImGui::SameLine();
            if (ImGui::ArrowButton("##RotX_Down", ImGuiDir_Down)) {
                diff_x = -step;
                rotation_changed = true;
                rotate_timer_x = 0.0f;
            }
            if (ImGui::IsItemActive()) {
                rotate_timer_x += ImGui::GetIO().DeltaTime;
                if (rotate_timer_x >= repeat_rate) {
                    diff_x = -step;
                    rotation_changed = true;
                    rotate_timer_x = 0.0f;
                }
            }

            // --- Y-axis ---
            ImGui::Text("Y-axis:");
            ImGui::SameLine();
            ImGui::Text("RotY");

            if (ImGui::ArrowButton("##RotY_Up", ImGuiDir_Up)) {
                diff_y = step;
                rotation_changed = true;
                rotate_timer_y = 0.0f;
            }
            if (ImGui::IsItemActive()) {
                rotate_timer_y += ImGui::GetIO().DeltaTime;
                if (rotate_timer_y >= repeat_rate) {
                    diff_y = step;
                    rotation_changed = true;
                    rotate_timer_y = 0.0f;
                }
            }

            ImGui::SameLine();
            if (ImGui::ArrowButton("##RotY_Down", ImGuiDir_Down)) {
                diff_y = -step;
                rotation_changed = true;
                rotate_timer_y = 0.0f;
            }
            if (ImGui::IsItemActive()) {
                rotate_timer_y += ImGui::GetIO().DeltaTime;
                if (rotate_timer_y >= repeat_rate) {
                    diff_y = -step;
                    rotation_changed = true;
                    rotate_timer_y = 0.0f;
                }
            }

            // --- Z-axis ---
            ImGui::Text("Z-axis:");
            ImGui::SameLine();
            ImGui::Text("RotZ");

            if (ImGui::ArrowButton("##RotZ_Up", ImGuiDir_Up)) {
                diff_z = step;
                rotation_changed = true;
                rotate_timer_z = 0.0f;
            }
            if (ImGui::IsItemActive()) {
                rotate_timer_z += ImGui::GetIO().DeltaTime;
                if (rotate_timer_z >= repeat_rate) {
                    diff_z = step;
                    rotation_changed = true;
                    rotate_timer_z = 0.0f;
                }
            }

            ImGui::SameLine();
            if (ImGui::ArrowButton("##RotZ_Down", ImGuiDir_Down)) {
                diff_z = -step;
                rotation_changed = true;
                rotate_timer_z = 0.0f;
            }
            if (ImGui::IsItemActive()) {
                rotate_timer_z += ImGui::GetIO().DeltaTime;
                if (rotate_timer_z >= repeat_rate) {
                    diff_z = -step;
                    rotation_changed = true;
                    rotate_timer_z = 0.0f;
                }
            }

            if (rotation_changed) {
                updateRotationMatrix(diff_x, diff_y, diff_z);
            }
        }

        void CropBoxPanel::renderBoundsControls() {
            if (ImGui::TreeNode("Bounds")) {
                glm::vec3 current_min = crop_box_->getMinBounds();
                glm::vec3 current_max = crop_box_->getMaxBounds();

                float min_bounds[3] = {current_min.x, current_min.y, current_min.z};
                float max_bounds[3] = {current_max.x, current_max.y, current_max.z};

                bool bounds_changed = false;

                renderMinBoundsInputs(min_bounds, max_bounds, bounds_changed);
                ImGui::Separator();
                renderMaxBoundsInputs(min_bounds, max_bounds, bounds_changed);

                if (bounds_changed) {
                    crop_box_->setBounds(
                        glm::vec3(min_bounds[0], min_bounds[1], min_bounds[2]),
                        glm::vec3(max_bounds[0], max_bounds[1], max_bounds[2]));
                }

                renderBoundsInfo();

                ImGui::TreePop();
            }
        }

        void CropBoxPanel::renderMinBoundsInputs(float min_bounds[3], float max_bounds[3], bool& bounds_changed) {
            const float min_range = -8.0f;
            const float max_range = 8.0f;
            const float bound_text_width = 110.0f;
            const float bound_step = 0.01f;
            const float bound_step_fast = 0.1f;

            ImGui::Text("Ctrl+click for faster steps");
            ImGui::Text("Min Bounds:");

            // Min X
            ImGui::Text("X:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(bound_text_width);
            float min_bounds_x = min_bounds[0];
            bounds_changed |= ImGui::InputFloat("##MinX", &min_bounds_x, bound_step, bound_step_fast, "%.3f");
            min_bounds_x = std::clamp(min_bounds_x, min_range, max_range);
            min_bounds_x = std::min(min_bounds_x, max_bounds[0]);
            min_bounds[0] = min_bounds_x;

            // Min Y
            ImGui::Text("Y:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(bound_text_width);
            float min_bounds_y = min_bounds[1];
            bounds_changed |= ImGui::InputFloat("##MinY", &min_bounds_y, bound_step, bound_step_fast, "%.3f");
            min_bounds_y = std::clamp(min_bounds_y, min_range, max_range);
            min_bounds_y = std::min(min_bounds_y, max_bounds[1]);
            min_bounds[1] = min_bounds_y;

            // Min Z
            ImGui::Text("Z:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(bound_text_width);
            float min_bounds_z = min_bounds[2];
            bounds_changed |= ImGui::InputFloat("##MinZ", &min_bounds_z, bound_step, bound_step_fast, "%.3f");
            min_bounds_z = std::clamp(min_bounds_z, min_range, max_range);
            min_bounds_z = std::min(min_bounds_z, max_bounds[2]);
            min_bounds[2] = min_bounds_z;
        }

        void CropBoxPanel::renderMaxBoundsInputs(float min_bounds[3], float max_bounds[3], bool& bounds_changed) {
            const float min_range = -8.0f;
            const float max_range = 8.0f;
            const float bound_text_width = 110.0f;
            const float bound_step = 0.01f;
            const float bound_step_fast = 0.1f;

            ImGui::Text("Max Bounds:");

            // Max X
            ImGui::Text("X:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(bound_text_width);
            float max_bounds_x = max_bounds[0];
            bounds_changed |= ImGui::InputFloat("##MaxX", &max_bounds_x, bound_step, bound_step_fast, "%.3f");
            max_bounds_x = std::clamp(max_bounds_x, min_range, max_range);
            max_bounds_x = std::max(max_bounds_x, min_bounds[0]);
            max_bounds[0] = max_bounds_x;

            // Max Y
            ImGui::Text("Y:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(bound_text_width);
            float max_bounds_y = max_bounds[1];
            bounds_changed |= ImGui::InputFloat("##MaxY", &max_bounds_y, bound_step, bound_step_fast, "%.3f");
            max_bounds_y = std::clamp(max_bounds_y, min_range, max_range);
            max_bounds_y = std::max(max_bounds_y, min_bounds[1]);
            max_bounds[1] = max_bounds_y;

            // Max Z
            ImGui::Text("Z:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(bound_text_width);
            float max_bounds_z = max_bounds[2];
            bounds_changed |= ImGui::InputFloat("##MaxZ", &max_bounds_z, bound_step, bound_step_fast, "%.3f");
            max_bounds_z = std::clamp(max_bounds_z, min_range, max_range);
            max_bounds_z = std::max(max_bounds_z, min_bounds[2]);
            max_bounds[2] = max_bounds_z;
        }

        void CropBoxPanel::renderBoundsInfo() {
            // Display current info
            glm::vec3 center = crop_box_->getCenter();
            glm::vec3 size = crop_box_->getSize();

            ImGui::Text("Center: (%.3f, %.3f, %.3f)", center.x, center.y, center.z);
            ImGui::Text("Size: (%.3f, %.3f, %.3f)", size.x, size.y, size.z);
        }

        // Helper function to wrap angles to 0-360 range
        float CropBoxPanel::wrapAngle(float angle) {
            while (angle < 0.0f)
                angle += 360.0f;
            while (angle >= 360.0f)
                angle -= 360.0f;
            return angle;
        }

        // Orthonormalize 3x3 rotation part of a 4x4 matrix
        glm::mat4 OrthonormalizeRotation(const glm::mat4& matrix) {
            glm::vec3 x = glm::vec3(matrix[0]);
            glm::vec3 y = glm::vec3(matrix[1]);
            glm::vec3 z = glm::vec3(matrix[2]);

            // Orthonormalize using Gram-Schmidt
            x = glm::normalize(x);
            y = glm::normalize(y - x * glm::dot(x, y));
            z = glm::normalize(glm::cross(x, y)); // Ensures right-handed coordinate system

            glm::mat4 result = glm::mat4(1.0f);
            result[0] = glm::vec4(x, 0.0f);
            result[1] = glm::vec4(y, 0.0f);
            result[2] = glm::vec4(z, 0.0f);
            result[3] = matrix[3]; // Preserve translation

            return result;
        }

        // Update the rotation matrix
        void CropBoxPanel::updateRotationMatrix(float delta_rot_x, float delta_rot_y, float delta_rot_z) {
            if (!crop_box_)
                return;

            // Convert degrees to radians
            float rad_x = glm::radians(delta_rot_x);
            float rad_y = glm::radians(delta_rot_y);
            float rad_z = glm::radians(delta_rot_z);

            // Create rotation matrices for each axis (in crop box coordinate system)
            glm::mat4 rot_x = glm::mat4(1.0f);
            rot_x[1][1] = cos(rad_x);
            rot_x[1][2] = sin(rad_x);
            rot_x[2][1] = -sin(rad_x);
            rot_x[2][2] = cos(rad_x);

            glm::mat4 rot_y = glm::mat4(1.0f);
            rot_y[0][0] = cos(rad_y);
            rot_y[0][2] = -sin(rad_y);
            rot_y[2][0] = sin(rad_y);
            rot_y[2][2] = cos(rad_y);

            glm::mat4 rot_z = glm::mat4(1.0f);
            rot_z[0][0] = cos(rad_z);
            rot_z[0][1] = sin(rad_z);
            rot_z[1][0] = -sin(rad_z);
            rot_z[1][1] = cos(rad_z);

            // Combine rotations: apply in order Z, Y, X (intrinsic rotations)
            glm::mat4 combined_rotation = rot_x * rot_y * rot_z;

            // Get the current center of the bounding box
            glm::vec3 center = crop_box_->getCenter();

            // Create translation matrices to move rotation center to origin and back
            glm::mat4 translate_to_origin = glm::mat4(1.0f);
            translate_to_origin[3][0] = -center.x;
            translate_to_origin[3][1] = -center.y;
            translate_to_origin[3][2] = -center.z;

            glm::mat4 translate_back = glm::mat4(1.0f);
            translate_back[3][0] = center.x;
            translate_back[3][1] = center.y;
            translate_back[3][2] = center.z;

            // Create the rotation transformation (translate to origin, rotate, translate back)
            glm::mat4 rotation_transform = translate_back * combined_rotation * translate_to_origin;

            glm::mat4 curr_world2bbox_ = crop_box_->getworld2BBox();
            // Apply the rotation transformation to the base transformation
            // The rotation happens in world space, so we compose: new_transform = rotation * base_transform
            glm::mat4 final_transform = rotation_transform * curr_world2bbox_;
            // Interesting problem: had to orthonormalize the matrix because of numerical issues
            // (multiplying rotations many times gets you out of the rotations group)
            final_transform = OrthonormalizeRotation(final_transform);

            // 2. Check orthonormal rotation: norm(R * R^T - I) < epsilon
            // Update the world2BBox transformation matrix
            crop_box_->setworld2BBox(final_transform);
        }

        // ============================================================================
        // GuiManager Implementation
        // ============================================================================

        GuiManager::GuiManager(GSViewer* viewer, std::shared_ptr<EventBus> event_bus)
            : viewer_(viewer),
              event_bus_(event_bus) {

            // Initialize components
            scripting_console_ = std::make_unique<ScriptingConsole>();
            file_browser_ = std::make_unique<FileBrowser>();
            camera_controls_ = std::make_unique<CameraControlsWindow>();
            training_controls_ = std::make_unique<TrainingControlsPanel>();
            crop_box_panel_ = std::make_unique<CropBoxPanel>();
            scene_panel_ = std::make_unique<ScenePanel>(*event_bus);

            // Set crop box reference
            if (viewer_ && viewer_->getCropBox()) {
                crop_box_panel_->crop_box_ = viewer_->getCropBox();
            }

            // Set up scene panel callbacks
            scene_panel_->setOnDatasetLoad([this](const std::filesystem::path& path) {
                if (path.empty()) {
                    // Empty path means open file browser
                    show_file_browser_ = true;
                } else {
                    // Non-empty path means load the dataset
                    file_browser_->setSelectedPath(path);
                    event_bus_->publish(LogMessageEvent{
                        LogMessageEvent::Level::Info,
                        std::format("Loading dataset from Scene Panel: {}", path.string()),
                        std::string("GuiManager")});
                    event_bus_->publish(LoadFileCommand{path, true});
                }
            });

            // Set up event handlers
            setupEventHandlers();
        }
        GuiManager::~GuiManager() {
            // Event handlers are automatically cleaned up when event bus is destroyed
        }

        void GuiManager::setupEventHandlers() {
            // Subscribe to events
            event_handler_ids_.push_back(
                event_bus_->subscribe<SceneLoadedEvent>(
                    [this](const SceneLoadedEvent& e) { handleSceneLoaded(e); }));

            event_handler_ids_.push_back(
                event_bus_->subscribe<SceneClearedEvent>(
                    [this](const SceneClearedEvent& e) { handleSceneCleared(e); }));

            event_handler_ids_.push_back(
                event_bus_->subscribe<TrainingStartedEvent>(
                    [this](const TrainingStartedEvent& e) { handleTrainingStarted(e); }));

            event_handler_ids_.push_back(
                event_bus_->subscribe<TrainingProgressEvent>(
                    [this](const TrainingProgressEvent& e) { handleTrainingProgress(e); }));

            event_handler_ids_.push_back(
                event_bus_->subscribe<TrainingPausedEvent>(
                    [this](const TrainingPausedEvent& e) { handleTrainingPaused(e); }));

            event_handler_ids_.push_back(
                event_bus_->subscribe<TrainingResumedEvent>(
                    [this](const TrainingResumedEvent& e) { handleTrainingResumed(e); }));

            event_handler_ids_.push_back(
                event_bus_->subscribe<TrainingCompletedEvent>(
                    [this](const TrainingCompletedEvent& e) { handleTrainingCompleted(e); }));

            event_handler_ids_.push_back(
                event_bus_->subscribe<LogMessageEvent>(
                    [this](const LogMessageEvent& e) { handleLogMessage(e); }));
        }

        void GuiManager::handleSceneLoaded(const SceneLoadedEvent& event) {
            scripting_console_->addLog("Info: Scene loaded from %s", event.source_path.string().c_str());
            scripting_console_->addLog("Info: Number of gaussians: %zu", event.num_gaussians);
            scripting_console_->addLog("Info: Source type: %s",
                                       event.source_type == SceneLoadedEvent::SourceType::PLY ? "PLY" : "Dataset");

            training_state_.training_started = false;
            training_state_.save_in_progress = false;
        }

        void GuiManager::handleSceneCleared(const SceneClearedEvent& event) {
            scripting_console_->addLog("Info: Scene cleared");
            training_state_.training_started = false;
            training_state_.save_in_progress = false;
        }

        void GuiManager::handleTrainingStarted(const TrainingStartedEvent& event) {
            training_state_.training_started = true;
            scripting_console_->addLog("Info: Training started (%d iterations)", event.total_iterations);
        }

        void GuiManager::handleTrainingProgress(const TrainingProgressEvent& event) {
            // Update progress display with throttling
            if (viewer_->info_) {
                viewer_->info_->updateProgress(event.iteration, event.total_iterations);
                viewer_->info_->updateNumSplats(event.num_gaussians);
                viewer_->info_->updateLoss(event.loss);
            }
        }

        void GuiManager::handleTrainingPaused(const TrainingPausedEvent& event) {
            scripting_console_->addLog("Info: Training paused at iteration %d", event.iteration);
        }

        void GuiManager::handleTrainingResumed(const TrainingResumedEvent& event) {
            scripting_console_->addLog("Info: Training resumed at iteration %d", event.iteration);
        }

        void GuiManager::handleTrainingCompleted(const TrainingCompletedEvent& event) {
            if (event.success) {
                scripting_console_->addLog("Info: Training completed successfully at iteration %d",
                                           event.final_iteration);
            } else {
                scripting_console_->addLog("Error: Training failed at iteration %d: %s",
                                           event.final_iteration,
                                           event.error_message.value_or("Unknown error").c_str());
            }
            training_state_.training_started = false;
        }

        void GuiManager::handleLogMessage(const LogMessageEvent& event) {
            const char* level_str = "";
            switch (event.level) {
            case LogMessageEvent::Level::Info: level_str = "Info"; break;
            case LogMessageEvent::Level::Warning: level_str = "Warning"; break;
            case LogMessageEvent::Level::Error: level_str = "Error"; break;
            case LogMessageEvent::Level::Debug: level_str = "Debug"; break;
            }

            if (event.source) {
                scripting_console_->addLog("%s [%s]: %s",
                                           level_str,
                                           event.source->c_str(),
                                           event.message.c_str());
            } else {
                scripting_console_->addLog("%s: %s", level_str, event.message.c_str());
            }
        }

        void GuiManager::init() {
            // Setup Dear ImGui context
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGuiIO& io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
            io.ConfigWindowsMoveFromTitleBarOnly = true;
            ImGui::StyleColorsLight();

            // Store GUI manager pointer in IO for access in callbacks
            io.UserData = this;

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

            if (show_scene_panel_) { // ADDED
                scene_panel_->render(&show_scene_panel_);
            }

            endFrame();
        }

        void GuiManager::renderMainPanel() {
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));
            ImGui::Begin("Rendering Setting", nullptr, window_flags_);
            ImGui::SetWindowSize(ImVec2(300, 0));

            // Scripting Console Button
            if (ImGui::Button("Open Scripting Console", ImVec2(-1, 0))) {
                show_scripting_console_ = true;
            }

            // Camera Controls Button
            if (ImGui::Button("Open Camera Controls", ImVec2(-1, 0))) {
                show_camera_controls_ = true;
            }

            ImGui::Separator();

            // Window toggles
            ImGui::Text("Windows");
            ImGui::Checkbox("Scripting Console", &show_scripting_console_);
            ImGui::Checkbox("Camera Controls", &show_camera_controls_);
            ImGui::Checkbox("Scene Panel", &show_scene_panel_);

            ImGui::Separator();

            // Mode status
            renderModeStatus();

            ImGui::Separator();

            // Rendering settings
            renderRenderingSettings();

            ImGui::Separator();

            // Training controls
            if (viewer_->getTrainer()) {
                training_controls_->render(viewer_->getTrainer(), training_state_, viewer_->notifier_);
            }

            ImGui::Separator();

            // Progress info
            renderProgressInfo();

            ImGui::Separator();

            // Crop box controls
            crop_box_panel_->render();

            ImGui::End();
            ImGui::PopStyleColor();
        }

        void GuiManager::renderModeStatus() {
            // Query current scene mode via events
            if (event_bus_) {
                EventResponseHandler<QuerySceneModeRequest, QuerySceneModeResponse> handler(event_bus_);
                auto response = handler.querySync(QuerySceneModeRequest{});

                if (response) {
                    switch (response->mode) {
                    case QuerySceneModeResponse::Mode::Empty:
                        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No data loaded");
                        ImGui::Text("Use File Browser to load:");
                        ImGui::BulletText("PLY file for viewing");
                        ImGui::BulletText("Dataset for training");
                        break;

                    case QuerySceneModeResponse::Mode::Viewing:
                        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "PLY Viewer Mode");
                        ImGui::Text("File: %s", viewer_->getCurrentPLYPath().filename().string().c_str());
                        break;

                    case QuerySceneModeResponse::Mode::Training:
                        ImGui::TextColored(ImVec4(0.2f, 0.5f, 0.8f, 1.0f), "Training Mode");
                        ImGui::Text("Dataset: %s", viewer_->getCurrentDatasetPath().filename().string().c_str());
                        break;
                    }
                }
            }
        }

        void GuiManager::renderRenderingSettings() {
            auto config = viewer_->getRenderingConfig();

            ImGui::Separator();
            ImGui::Text("Rendering Settings");
            ImGui::Separator();

            float old_scale = config->scaling_modifier;
            ImGui::SetNextItemWidth(200);
            ImGui::SliderFloat("##scale_slider", &config->scaling_modifier, 0.01f, 3.0f, "Scale=%.2f");
            if (old_scale != config->scaling_modifier) {
                publish(RenderingSettingsChangedEvent{
                    std::nullopt, config->scaling_modifier, std::nullopt});
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset##scale", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
                config->scaling_modifier = 1.0f;
                publish(RenderingSettingsChangedEvent{
                    std::nullopt, config->scaling_modifier, std::nullopt});
            }

            float old_fov = config->fov;
            ImGui::SetNextItemWidth(200);
            ImGui::SliderFloat("##fov_slider", &config->fov, 45.0f, 120.0f, "FoV=%.2f");
            if (old_fov != config->fov) {
                publish(RenderingSettingsChangedEvent{
                    config->fov, std::nullopt, std::nullopt});
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset##fov", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
                config->fov = 75.0f;
                publish(RenderingSettingsChangedEvent{
                    config->fov, std::nullopt, std::nullopt});
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
            auto trainer_manager = viewer_->getTrainerManager();

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

            // Use ImGui's built-in frame rate to naturally throttle updates
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

        void GuiManager::showScenePanel(bool show) {
            show_scene_panel_ = show;
        }

        void GuiManager::setScriptExecutor(std::function<std::string(const std::string&)> executor) {
            scripting_console_->setExecutor(executor);
        }

        void GuiManager::setFileSelectedCallback(std::function<void(const std::filesystem::path&, bool)> callback) {
            file_browser_->setOnFileSelected([this](const std::filesystem::path& path, bool is_dataset) {
                publish(LoadFileCommand{path, is_dataset});
            });
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

    } // namespace gui
} // namespace gs