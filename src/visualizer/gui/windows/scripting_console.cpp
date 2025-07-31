#include "gui/windows/scripting_console.hpp"
#include <cstdarg>
#include <cstdio>
#include <imgui.h>

namespace gs::gui {

    ScriptingConsole::ScriptingConsole() {
        clearLog();
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

        history_.push_back(command);

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

        // Add NoDocking flag to prevent this window from being docked
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.08f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.1f, 0.1f, 0.15f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.15f, 0.15f, 0.25f, 1.0f));

        if (!ImGui::Begin("Scripting Console", p_open, window_flags)) {
            ImGui::End();
            ImGui::PopStyleColor(4);
            return;
        }

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

        const float footer_height_to_reserve = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
        if (ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footer_height_to_reserve), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));

            for (const auto& line : output_buffer_) {
                ImVec4 color;
                bool has_color = false;

                if (line.find(">>>") == 0) {
                    color = ImVec4(0.8f, 0.8f, 0.2f, 1.0f);
                    has_color = true;
                } else if (line.find("Error:") == 0) {
                    color = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
                    has_color = true;
                } else if (line.find("Info:") == 0 || line.find("GPU Memory") == 0 ||
                           line.find("Model Information") == 0 || line.find("Training Status") == 0) {
                    color = ImVec4(0.3f, 0.8f, 0.3f, 1.0f);
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

        ImGui::Separator();

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

        ImGui::SetItemDefaultFocus();
        if (reclaim_focus)
            ImGui::SetKeyboardFocusHere(-1);

        ImGui::End();
        ImGui::PopStyleColor(4);
    }

    void ScriptingConsole::setExecutor(std::function<std::string(const std::string&)> executor) {
        execute_callback_ = executor;
    }
} // namespace gs::gui
