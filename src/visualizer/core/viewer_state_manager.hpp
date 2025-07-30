#pragma once

#include "core/events.hpp"
#include <atomic>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>

namespace gs::visualizer {

    // Move TrainingInfo out of VisualizerImpl
    struct TrainingInfo {
        std::atomic<int> curr_iterations_{0};
        std::atomic<int> total_iterations_{0};
        std::atomic<int> num_splats_{0};

        int max_loss_points_ = 200;
        std::deque<float> loss_buffer_;
        std::mutex loss_buffer_mutex_;

        void updateProgress(int iter, int total_iterations);
        void updateNumSplats(int num_splats);
        void updateLoss(float loss);
        std::deque<float> getLossBuffer() const;
    };

    // Move RenderingConfig out of VisualizerImpl
    struct RenderingConfig {
        mutable std::mutex mtx;
        float fov = 60.0f;
        float scaling_modifier = 1.0f;

        glm::vec2 getFov(size_t reso_x, size_t reso_y) const;
        float getFovDegrees() const;
        float getScalingModifier() const;
        void setFov(float f);
        void setScalingModifier(float s);
    };

    enum class ViewerMode {
        Empty,
        PLYViewer,
        Training
    };

    class ViewerStateManager {
    public:
        ViewerStateManager();
        ~ViewerStateManager();

        // Mode management
        ViewerMode getCurrentMode() const { return current_mode_; }
        void setMode(ViewerMode mode);

        // Path management
        const std::filesystem::path& getCurrentPLYPath() const { return current_ply_path_; }
        const std::filesystem::path& getCurrentDatasetPath() const { return current_dataset_path_; }
        void setPLYPath(const std::filesystem::path& path);
        void setDatasetPath(const std::filesystem::path& path);
        void clearPaths();

        // State components
        std::shared_ptr<TrainingInfo> getTrainingInfo() { return training_info_; }
        std::shared_ptr<RenderingConfig> getRenderingConfig() { return rendering_config_; }

        // Rendering state
        bool isAntiAliasingEnabled() const { return anti_aliasing_; }
        void setAntiAliasing(bool enabled);

        // Clear all state
        void reset();

    private:
        void setupEventHandlers();
        void publishStateChange();

        // Current state
        std::atomic<ViewerMode> current_mode_{ViewerMode::Empty};
        std::filesystem::path current_ply_path_;
        std::filesystem::path current_dataset_path_;

        // State components
        std::shared_ptr<TrainingInfo> training_info_;
        std::shared_ptr<RenderingConfig> rendering_config_;

        // Rendering state
        std::atomic<bool> anti_aliasing_{false};

        mutable std::mutex paths_mutex_;
    };

} // namespace gs::visualizer
