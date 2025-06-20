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
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <chrono>
#include <condition_variable>
#include <cuda_runtime.h>
#include <deque>
#include <glm/glm.hpp>
#include <imgui.h>
#include <iostream>
#include <memory>
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

    public:
        GSViewer(std::string title, int width, int height);
        ~GSViewer();

        void setTrainer(Trainer* trainer);

        void drawFrame();

        void configuration();

        void draw() override;

    public:
        std::shared_ptr<TrainingInfo> info_;

        std::shared_ptr<Notifier> notifier_;

        std::mutex splat_mtx_;

    private:
        std::shared_ptr<RenderingConfig> config_;

        Trainer* trainer_;

        // Control button states
        bool show_control_panel_ = true;
        bool save_in_progress_ = false;
        std::chrono::steady_clock::time_point save_start_time_;
        bool manual_start_triggered_ = false;
        bool training_started_ = false;
    };

} // namespace gs