#pragma once

#include "core/bilateral_grid.hpp"
#include "core/dataset.hpp"
#include "core/events.hpp"
#include "core/istrategy.hpp"
#include "core/metrics.hpp"
#include "core/parameters.hpp"
#include "core/training_progress.hpp"
#include <atomic>
#include <expected>
#include <memory>
#include <shared_mutex>
#include <stop_token>
#include <torch/torch.h>

// Forward declaration
class Camera;

namespace gs {

    class Trainer {
    public:
        // Constructor that takes ownership of strategy and shares datasets
        Trainer(std::shared_ptr<CameraDataset> dataset,
                std::unique_ptr<IStrategy> strategy,
                const param::TrainingParameters& params);

        // Delete copy operations
        Trainer(const Trainer&) = delete;
        Trainer& operator=(const Trainer&) = delete;

        // Allow move operations
        Trainer(Trainer&&) = default;
        Trainer& operator=(Trainer&&) = default;

        ~Trainer();

        // Main training method with stop token support
        std::expected<void, std::string> train(std::stop_token stop_token = {});

        // Control methods for GUI interaction
        void request_pause() { pause_requested_ = true; }
        void request_resume() { pause_requested_ = false; }
        void request_save() { save_requested_ = true; }
        void request_stop() { stop_requested_ = true; } // This will fully stop training

        bool is_paused() const { return is_paused_.load(); }
        bool is_running() const { return is_running_.load(); }
        bool is_training_complete() const { return training_complete_.load(); }
        bool has_stopped() const { return stop_requested_.load(); } // Check if stop was requested

        // Get current training state
        int get_current_iteration() const { return current_iteration_.load(); }
        float get_current_loss() const { return current_loss_.load(); }

        // just for viewer to get model
        const IStrategy& get_strategy() const { return *strategy_; }

        // Allow viewer to lock for rendering
        std::shared_mutex& getRenderMutex() const { return render_mutex_; }

        const param::TrainingParameters& getParams() const { return params_; }

    private:
        // Training step result
        enum class StepResult {
            Continue,
            Stop,
            Error
        };

        // Protected method for processing a single training step
        // Returns result indicating whether training should continue
        std::expected<StepResult, std::string> train_step(
            int iter,
            Camera* cam, // Use global Camera, not gs::Camera
            torch::Tensor gt_image,
            torch::Tensor weights,
            RenderMode render_mode,
            float out_of_mask_penalty,
            std::stop_token stop_token = {});

        // Protected methods for computing loss - now return expected values
        std::expected<torch::Tensor, std::string> compute_photometric_loss(
            const RenderOutput& render_output,
            const torch::Tensor& gt_image,
            const SplatData& splatData,
            const param::OptimizationParameters& opt_params);
            
        std::expected<torch::Tensor, std::string> compute_photometric_loss(
            const RenderOutput& render_output,
            const torch::Tensor& gt_image,
            const torch::Tensor& weights,
            const float outOfMaskAlphaPenalty,
            const SplatData& splatData,
            const param::OptimizationParameters& opt_params);

        std::expected<torch::Tensor, std::string> compute_scale_reg_loss(
            const SplatData& splatData,
            const param::OptimizationParameters& opt_params);

        std::expected<torch::Tensor, std::string> compute_opacity_reg_loss(
            const SplatData& splatData,
            const param::OptimizationParameters& opt_params);

        std::expected<torch::Tensor, std::string> compute_bilateral_grid_tv_loss(
            const std::unique_ptr<gs::BilateralGrid>& bilateral_grid,
            const param::OptimizationParameters& opt_params);

        std::expected<void, std::string> initialize_bilateral_grid();

        // Handle control requests
        void handle_control_requests(int iter, std::stop_token stop_token = {});

        // Prune gaussians using masks
        void prune_after_training(float threshold);
        
        // Member variables
        std::shared_ptr<CameraDataset> train_dataset_;
        std::shared_ptr<CameraDataset> val_dataset_;
        std::unique_ptr<IStrategy> strategy_;
        param::TrainingParameters params_;

        torch::Tensor background_{};
        std::unique_ptr<TrainingProgress> progress_;
        size_t train_dataset_size_;

        // Bilateral grid components
        std::unique_ptr<gs::BilateralGrid> bilateral_grid_;
        std::unique_ptr<torch::optim::Adam> bilateral_grid_optimizer_;

        // Metrics evaluator - handles all evaluation logic
        std::unique_ptr<metrics::MetricsEvaluator> evaluator_;

        // Single mutex that protects the model during training
        mutable std::shared_mutex render_mutex_;

        // Control flags for thread communication
        std::atomic<bool> pause_requested_{false};
        std::atomic<bool> save_requested_{false};
        std::atomic<bool> stop_requested_{false};
        std::atomic<bool> is_paused_{false};
        std::atomic<bool> is_running_{false};
        std::atomic<bool> training_complete_{false};

        // Current training state
        std::atomic<int> current_iteration_{0};
        std::atomic<float> current_loss_{0.0f};
    };

} // namespace gs