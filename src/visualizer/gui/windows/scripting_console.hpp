#pragma once

#include <functional>
#include <string>
#include <vector>

struct ImGuiInputTextCallbackData;

namespace gs::gui {

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

} // namespace gs::gui
