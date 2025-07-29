#pragma once

#include "core/error_handler.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/memory_monitor.hpp"
#include "core/parameters.hpp"
#include "gui/gui_manager.hpp"
#include "input/camera_controller.hpp"
#include "input/input_handler.hpp"
#include "internal/viewer_notifier.hpp"
#include "internal/viewport.hpp"
#include "rendering/render_bounding_box.hpp"
#include "rendering/renderer.hpp"
#include "scene/scene.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"
#include "visualizer/visualizer.hpp"
#include "window/window_manager.hpp"
#include <chrono>
#include <deque>
#include <memory>
#include <string>

namespace gs::visualizer {

    class VisualizerImpl : public Visualizer {
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
                while (loss_buffer_.size() > static_cast<size_t>(max_loss_points_)) {
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
        explicit VisualizerImpl(const ViewerOptions& options);
        ~VisualizerImpl() override;

        void run() override;
        void setParameters(const param::TrainingParameters& params) override;
        std::expected<void, std::string> loadPLY(const std::filesystem::path& path) override;
        std::expected<void, std::string> loadDataset(const std::filesystem::path& path) override;
        void clearScene() override;

        // Getters for GUI (needed because gui_manager expects these)
        ViewerMode getCurrentMode() const;
        Trainer* getTrainer() const { return trainer_manager_->getTrainer(); }
        std::shared_ptr<TrainingInfo> getTrainingInfo() const { return info_; }
        std::shared_ptr<RenderingConfig> getRenderingConfig() const { return config_; }
        std::shared_ptr<ViewerNotifier> getNotifier() const { return notifier_; }
        const std::filesystem::path& getCurrentPLYPath() const { return current_ply_path_; }
        const std::filesystem::path& getCurrentDatasetPath() const { return current_dataset_path_; }
        TrainerManager* getTrainerManager() { return trainer_manager_.get(); } // MOVED TO PUBLIC
        GLFWwindow* getWindow() const { return window_manager_->getWindow(); }
        std::shared_ptr<EventBus> getEventBus() const { return event_bus_; }
        std::shared_ptr<RenderBoundingBox> getCropBox() const { return crop_box_; }

        // Public members accessed by GUI
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

    private:
        // Initialization
        bool init();
        void updateWindowSize();
        float getGPUUsage();
        void setFrameRate(const int fps);
        void controlFrameRate();

        // Rendering
        void draw();
        void drawFrame();

        // Data loading
        void loadPLYFile(const std::filesystem::path& path);
        void loadDatasetInternal(const std::filesystem::path& path);
        void clearCurrentData();

        // Training control
        void startTraining();

        // GUI access
        bool isGuiActive() const;

        // Input handlers
        bool handleFileDrop(const InputHandler::FileDropEvent& event);

        // Event system
        void setupEventHandlers();
        void handleStartTrainingCommand(const StartTrainingCommand& cmd);
        void handlePauseTrainingCommand(const PauseTrainingCommand& cmd);
        void handleResumeTrainingCommand(const ResumeTrainingCommand& cmd);
        void handleStopTrainingCommand(const StopTrainingCommand& cmd);
        void handleSaveCheckpointCommand(const SaveCheckpointCommand& cmd);
        void handleLoadFileCommand(const LoadFileCommand& cmd);
        void handleRenderingSettingsChanged(const RenderingSettingsChangedEvent& event);

    private:
        // Options and parameters
        ViewerOptions options_;
        std::string title_;

        // Window and rendering
        Viewport viewport_;
        std::unique_ptr<WindowManager> window_manager_;
        std::unique_ptr<InputHandler> input_handler_;
        std::unique_ptr<CameraController> camera_controller_;
        std::shared_ptr<ScreenQuadRenderer> screen_renderer_;
        std::shared_ptr<Shader> quad_shader_;

        // Frame rate control
        int target_fps_ = 30;
        int frame_time_;
        std::chrono::time_point<std::chrono::high_resolution_clock> last_time_;

        // Event system
        std::shared_ptr<EventBus> event_bus_;
        std::unique_ptr<SceneManager> scene_manager_;
        std::vector<size_t> event_handler_ids_;

        // Error handling and monitoring
        std::unique_ptr<ErrorHandler> error_handler_;
        std::unique_ptr<MemoryMonitor> memory_monitor_;

        // Cache for last memory usage
        MemoryUsageEvent last_memory_usage_;

        // Bounding box visualization
        std::shared_ptr<RenderBoundingBox> crop_box_;
    };

} // namespace gs::visualizer