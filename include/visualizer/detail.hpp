#pragma once

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/rasterizer.hpp"
#include "core/trainer.hpp"
#include "visualizer/renderer.hpp"
// clang-format off
// CRITICAL: GLAD must be included before GLFW to avoid OpenGL header conflicts
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <chrono>
#include <condition_variable>
#include <cuda_runtime.h>
#include <deque>
#include <functional>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

using uchar = unsigned char;

namespace gs {

    class ViewerDetail {

    public:
        ViewerDetail(std::string title, int width, int height);

        ~ViewerDetail();

        bool init();

        void updateWindowSize();

        float getGPUUsage();

        void setFrameRate(const int fps);

        void controlFrameRate();

        static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

        static void cursorPosCallback(GLFWwindow* window, double x, double y);

        static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

        static void wsad_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

        void run();

        virtual void draw() = 0;

    protected:
        ImGuiWindowFlags window_flags = 0;

        bool any_window_active = false;

        Viewport viewport_;

        std::shared_ptr<ScreenQuadRenderer> screen_renderer_;

        std::shared_ptr<Shader> quadShader_;

    private:
        std::string title_;

        GLFWwindow* window_;

        static ViewerDetail* detail_;

        int targetFPS = 30;

        int frameTime;

        std::chrono::time_point<std::chrono::high_resolution_clock> lastTime;
    };

    class GSViewer : public ViewerDetail {

        struct Notifier {
        public:
            bool ready = false;
            std::mutex mtx;
            std::condition_variable cv;
        };

        struct TrainingInfo {

            std::mutex mtx;

            int curr_iterations_ = 0;
            int total_iterations_ = 0;

            int num_splats_ = 0;
            int max_loss_points_ = 200;

            std::deque<float> loss_buffer_;

            void updateProgress(int iter, int total_iterations) {
                curr_iterations_ = iter;
                total_iterations_ = total_iterations;
            }

            void updateNumSplats(int num_splats) {
                num_splats_ = num_splats;
            }

            void updateLoss(float loss) {
                loss_buffer_.push_back(loss);
                while (loss_buffer_.size() > max_loss_points_) {
                    loss_buffer_.pop_front();
                }
            }
        };

        struct RenderingConfig {

            std::mutex mtx;

            float fov = 60.0f;
            float scaling_modifier = 1.0f;

            glm::vec2 getFov(size_t reso_x, size_t reso_y) const {
                return glm::vec2(
                    atan(tan(glm::radians(fov) / 2.0f) * reso_x / reso_y) * 2.0f,
                    glm::radians(fov));
            }
        };

        struct ScriptingConsole {
            std::vector<std::string> history_;
            std::vector<std::string> output_buffer_;
            char input_buffer_[1024] = "";
            int history_pos_ = -1;
            bool scroll_to_bottom_ = false;
            bool reclaim_focus_ = false;
            size_t max_output_lines_ = 1000;

            // Callback function for executing scripts
            std::function<std::string(const std::string&)> execute_callback_;

            ScriptingConsole() {
                clearLog();
                // Set default callback that just echoes input
                execute_callback_ = [](const std::string& input) -> std::string {
                    return "Echo: " + input;
                };
            }

            void clearLog() {
                output_buffer_.clear();
            }

            void addLog(const char* fmt, ...) {
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

            void executeCommand(const std::string& command) {
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

            static int textEditCallbackStub(ImGuiInputTextCallbackData* data) {
                ScriptingConsole* console = (ScriptingConsole*)data->UserData;
                return console->textEditCallback(data);
            }

            int textEditCallback(ImGuiInputTextCallbackData* data) {
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
        };

    public:
        GSViewer(std::string title, int width, int height);
        ~GSViewer();

        void setTrainer(Trainer* trainer);
        void setStandaloneModel(std::unique_ptr<SplatData> model);
        void setAntiAliasing(bool enable);

        void drawFrame();

        void configuration();

        void draw() override;

        // Scripting system methods
        void renderScriptingConsole();
        void setScriptExecutor(std::function<std::string(const std::string&)> executor);

        // Add new file browser related methods
        void renderFileBrowser();
        void loadPLYFile(const std::filesystem::path& path);
        void loadDataset(const std::filesystem::path& path);
        void clearCurrentData();

        // Add mode enum
        enum class ViewerMode {
            Empty,     // No data loaded
            PLYViewer, // Viewing a PLY file
            Training   // Ready to train or training
        };

        ViewerMode getCurrentMode() const { return current_mode_; }

    public:
        std::shared_ptr<TrainingInfo> info_;

        std::shared_ptr<Notifier> notifier_;

        std::mutex splat_mtx_;

    private:
        std::shared_ptr<RenderingConfig> config_;

        Trainer* trainer_;
        std::unique_ptr<SplatData> standalone_model_;

        // Control button states
        bool show_control_panel_ = true;
        bool save_in_progress_ = false;
        std::chrono::steady_clock::time_point save_start_time_;
        bool manual_start_triggered_ = false;
        bool training_started_ = false;

        // camera controls
        void renderCameraControlsWindow();
        bool show_camera_controls_window_ = false;
        bool anti_aliasing_ = false;

        // Scripting console
        std::unique_ptr<ScriptingConsole> scripting_console_;
        bool show_scripting_console_ = false;

        // File browser state
        bool show_file_browser_ = false;
        std::string file_browser_current_path_;
        std::string file_browser_selected_file_;

        // Current mode
        ViewerMode current_mode_ = ViewerMode::Empty;

        // Store paths for current data
        std::filesystem::path current_ply_path_;
        std::filesystem::path current_dataset_path_;
    };

} // namespace gs