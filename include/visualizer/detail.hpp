#pragma once

#include "core/image_io.hpp"
#include "visualizer/training_manager.hpp"
#include "visualizer/camera_controller.hpp"
#include "visualizer/input_handler.hpp"
#include "visualizer/renderer.hpp"
#include "visualizer/scene.hpp"
#include "visualizer/viewer_notifier.hpp"
#include "visualizer/window_manager.hpp"
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <torch/torch.h>

using uchar = unsigned char;

namespace gs {

    // Forward declarations
    namespace gui {
        class GuiManager;
    }

    class ViewerDetail {

    public:
        ViewerDetail(std::string title, int width, int height);

        ~ViewerDetail();

        bool init();

        void updateWindowSize();

        float getGPUUsage();

        void setFrameRate(const int fps);

        void controlFrameRate();

        void run();

        virtual void draw() = 0;

        GLFWwindow* getWindow() const { return window_manager_->getWindow(); }

    protected:
        Viewport viewport_;

        std::shared_ptr<ScreenQuadRenderer> screen_renderer_;

        std::shared_ptr<Shader> quadShader_;

        std::unique_ptr<WindowManager> window_manager_;

        std::unique_ptr<InputHandler> input_handler_;

        std::unique_ptr<CameraController> camera_controller_;

    private:
        std::string title_;

        int targetFPS = 30;

        int frameTime;

        std::chrono::time_point<std::chrono::high_resolution_clock> lastTime;
    };

    class GSViewer : public ViewerDetail {

    public:
        struct TrainingInfo {

            // No mutex needed - use atomics
            std::atomic<int> curr_iterations_{0};
            std::atomic<int> total_iterations_{0};
            std::atomic<int> num_splats_{0};

            int max_loss_points_ = 200;
            std::deque<float> loss_buffer_;
            std::mutex loss_buffer_mutex_; // Only for loss buffer

            void updateProgress(int iter, int total_iterations) {
                curr_iterations_ = iter;
                total_iterations_ = total_iterations;
            }

            void updateNumSplats(int num_splats) {
                num_splats_ = num_splats;
            }

            void updateLoss(float loss) {
                std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
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

        // Add mode enum
        enum class ViewerMode {
            Empty,     // No data loaded
            PLYViewer, // Viewing a PLY file
            Training   // Ready to train or training
        };

    public:
        GSViewer(std::string title, int width, int height);
        ~GSViewer();

        void setTrainer(Trainer* trainer);
        void setStandaloneModel(std::unique_ptr<SplatData> model);
        void setAntiAliasing(bool enable);
        void setParameters(const gs::param::TrainingParameters& params) { params_ = params; }

        void drawFrame();

        void draw() override;

        // Data loading methods
        void loadPLYFile(const std::filesystem::path& path);
        void loadDataset(const std::filesystem::path& path);
        void clearCurrentData();

        // Getters for GUI
        ViewerMode getCurrentMode() const;
        Trainer* getTrainer() const { return trainer_manager_->getTrainer(); }
        SplatData* getStandaloneModel() const { return scene_->getStandaloneModel(); }
        std::shared_ptr<TrainingInfo> getTrainingInfo() const { return info_; }
        std::shared_ptr<RenderingConfig> getRenderingConfig() const { return config_; }
        std::shared_ptr<ViewerNotifier> getNotifier() const { return notifier_; }
        const std::filesystem::path& getCurrentPLYPath() const { return current_ply_path_; }
        const std::filesystem::path& getCurrentDatasetPath() const { return current_dataset_path_; }
        TrainerManager* getTrainerManager() { return trainer_manager_.get(); }

        // Training control (delegates to TrainerManager)
        void startTraining();

        // GUI access for static callbacks
        bool isGuiActive() const;

    private:
        // Input handlers
        bool handleFileDrop(const InputHandler::FileDropEvent& event);

    public:
        std::shared_ptr<TrainingInfo> info_;

        std::shared_ptr<ViewerNotifier> notifier_;

        std::unique_ptr<Scene> scene_;
        std::shared_ptr<RenderingConfig> config_;

        bool anti_aliasing_ = false;

        param::TrainingParameters params_;

        // Current mode
        ViewerMode current_mode_ = ViewerMode::Empty;

        // Store paths for current data
        std::filesystem::path current_ply_path_;
        std::filesystem::path current_dataset_path_;

        // Training management
        std::unique_ptr<TrainerManager> trainer_manager_;

        // GUI manager
        std::unique_ptr<gui::GuiManager> gui_manager_;
        friend class gui::GuiManager; // Allow GUI manager to access private members
    };

} // namespace gs