#include "core/viewer_state_manager.hpp"
#include <algorithm>

namespace gs::visualizer {

    // TrainingInfo implementation
    void TrainingInfo::updateProgress(int iter, int total_iterations) {
        curr_iterations_ = iter;
        total_iterations_ = total_iterations;
    }

    void TrainingInfo::updateNumSplats(int num_splats) {
        num_splats_ = num_splats;
    }

    void TrainingInfo::updateLoss(float loss) {
        std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
        loss_buffer_.push_back(loss);
        while (loss_buffer_.size() > static_cast<size_t>(max_loss_points_)) {
            loss_buffer_.pop_front();
        }
    }

    std::deque<float> TrainingInfo::getLossBuffer() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(loss_buffer_mutex_)); // Fix: cast away const
        return loss_buffer_;
    }

    // RenderingConfig implementation
    glm::vec2 RenderingConfig::getFov(size_t reso_x, size_t reso_y) const {
        std::lock_guard<std::mutex> lock(mtx);
        return glm::vec2(
            atan(tan(glm::radians(fov) / 2.0f) * reso_x / reso_y) * 2.0f,
            glm::radians(fov));
    }

    float RenderingConfig::getFovDegrees() const {
        std::lock_guard<std::mutex> lock(mtx);
        return fov;
    }

    float RenderingConfig::getScalingModifier() const {
        std::lock_guard<std::mutex> lock(mtx);
        return scaling_modifier;
    }

    void RenderingConfig::setFov(float f) {
        std::lock_guard<std::mutex> lock(mtx);
        fov = f;
    }

    void RenderingConfig::setScalingModifier(float s) {
        std::lock_guard<std::mutex> lock(mtx);
        scaling_modifier = s;
    }

    // ViewerStateManager implementation
    ViewerStateManager::ViewerStateManager() {
        training_info_ = std::make_shared<TrainingInfo>();
        rendering_config_ = std::make_shared<RenderingConfig>();
        setupEventHandlers();
    }

    ViewerStateManager::~ViewerStateManager() = default;

    void ViewerStateManager::setupEventHandlers() {
        using namespace events;

        // Listen for render settings changes
        ui::RenderSettingsChanged::when([this](const auto& event) {
            if (event.fov) {
                rendering_config_->setFov(*event.fov);
            }
            if (event.scaling_modifier) {
                rendering_config_->setScalingModifier(*event.scaling_modifier);
            }
            if (event.antialiasing) {
                setAntiAliasing(*event.antialiasing);
            }
        });

        // Listen for training progress
        state::TrainingProgress::when([this](const auto& event) {
            training_info_->updateProgress(event.iteration, training_info_->total_iterations_);
            training_info_->updateNumSplats(event.num_gaussians);
            training_info_->updateLoss(event.loss);
        });

        state::TrainingStarted::when([this](const auto& event) {
            training_info_->total_iterations_ = event.total_iterations;
        });
    }

    void ViewerStateManager::setMode(ViewerMode mode) {
        ViewerMode old_mode = current_mode_.exchange(mode);
        if (old_mode != mode) {
            publishStateChange();
        }
    }

    void ViewerStateManager::setPLYPath(const std::filesystem::path& path) {
        std::lock_guard<std::mutex> lock(paths_mutex_);
        current_ply_path_ = path;
        setMode(ViewerMode::PLYViewer);
    }

    void ViewerStateManager::setDatasetPath(const std::filesystem::path& path) {
        std::lock_guard<std::mutex> lock(paths_mutex_);
        current_dataset_path_ = path;
        setMode(ViewerMode::Training);
    }

    void ViewerStateManager::clearPaths() {
        std::lock_guard<std::mutex> lock(paths_mutex_);
        current_ply_path_.clear();
        current_dataset_path_.clear();
        setMode(ViewerMode::Empty);
    }

    void ViewerStateManager::setAntiAliasing(bool enabled) {
        anti_aliasing_ = enabled;
    }

    void ViewerStateManager::reset() {
        clearPaths();
        training_info_->curr_iterations_ = 0;
        training_info_->total_iterations_ = 0;
        training_info_->num_splats_ = 0;
        training_info_->loss_buffer_.clear();
    }

    void ViewerStateManager::publishStateChange() {
        // Since ViewerModeChanged doesn't exist, we can use a generic state change event
        // or create a custom event if needed. For now, let's emit a log event
        events::notify::Log{
            .level = events::notify::Log::Level::Debug,
            .message = "Viewer mode changed",
            .source = "ViewerStateManager"}
            .emit();
    }

} // namespace gs::visualizer