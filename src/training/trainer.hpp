/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "dataset.hpp"
#include "metrics/metrics.hpp"
#include "progress.hpp"
#include "project/project.hpp"
#include "rasterization/rasterizer.hpp"
#include "strategies/istrategy.hpp"
#include <atomic>
#include <c10/cuda/CUDAStream.h>
#include <expected>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stop_token>
#include <thread>
#include <unordered_map>

namespace gs {
    namespace management {
        class Project;
    }

    namespace training {
        // Forward declarations
        class BilateralGrid;
        class PoseOptimizationModule;
        class WarmupExponentialLR;
        class ISparsityOptimizer;
        struct RenderOutput;

        // Deferred event emission helper
        class DeferredEvents {
            std::vector<std::function<void()>> events_;

        public:
            template <typename Event>
            void add(Event&& event) {
                events_.push_back([e = std::forward<Event>(event)]() { e.emit(); });
            }
            ~DeferredEvents() {
                for (auto& emit_fn : events_) {
                    emit_fn();
                }
            }
        };

        class Trainer {
        public:
            // Construction
            Trainer(std::shared_ptr<CameraDataset> dataset,
                    std::unique_ptr<IStrategy> strategy);
            ~Trainer();

            // Delete copy
            Trainer(const Trainer&) = delete;
            Trainer& operator=(const Trainer&) = delete;

            // Allow move
            Trainer(Trainer&&) = default;
            Trainer& operator=(Trainer&&) = default;

            // Initialize with parameters
            std::expected<void, std::string> initialize(const param::TrainingParameters& params);

            // Main training loop
            std::expected<void, std::string> train(std::stop_token stop_token = {});

            // Control methods
            void pause() { pause_requested_ = true; }
            void resume() { pause_requested_ = false; }
            void stop() { stop_requested_ = true; }
            void request_save() { save_requested_ = true; }

            // Status getters
            bool is_running() const { return is_running_.load(); }
            bool is_paused() const { return is_paused_.load(); }
            bool is_training_complete() const { return training_complete_.load(); }
            int get_current_iteration() const { return current_iteration_.load(); }
            float get_current_loss() const { return current_loss_.load(); }

            // Model access (thread-safe)
            const SplatData& getModel() const {
                std::shared_lock<std::shared_mutex> lock(render_mutex_);
                return strategy_->get_model();
            }
            SplatData& getModel() {
                std::unique_lock<std::shared_mutex> lock(render_mutex_);
                return strategy_->get_model();
            }

            // Camera access
            std::shared_ptr<const Camera> getCamById(int camId) const;
            std::vector<std::shared_ptr<const Camera>> getCamList() const;

            // Progress callback
            void set_progress_callback(std::function<void()> callback) {
                callback_ = std::move(callback);
            }

            // Project management
            void setProject(std::shared_ptr<gs::management::Project> project) {
                lf_project_ = project;
            }

            // Dataset info
            std::shared_ptr<CameraDataset> get_dataset() const { return train_dataset_; }

        private:
            // Training step result
            enum class StepResult {
                Continue,
                Stop
            };

            // Helper methods
            void cleanup();
            void load_cameras_info();
            void handle_control_requests(int iter, std::stop_token stop_token);
            void save_ply(const std::filesystem::path& save_path, int iter_num, bool join_threads = true);

            // Initialization helpers
            std::expected<void, std::string> initialize_bilateral_grid();

            // Loss computation helpers - Updated to use StrategyParameters
            std::expected<torch::Tensor, std::string> compute_photometric_loss(
                const RenderOutput& render_output,
                const torch::Tensor& gt_image,
                const SplatData& splatData,
                const param::StrategyParameters& opt_params);

            std::expected<torch::Tensor, std::string> compute_scale_reg_loss(
                const SplatData& splatData,
                const param::StrategyParameters& opt_params);

            std::expected<torch::Tensor, std::string> compute_opacity_reg_loss(
                const SplatData& splatData,
                const param::StrategyParameters& opt_params);

            std::expected<torch::Tensor, std::string> compute_bilateral_grid_tv_loss(
                const std::unique_ptr<BilateralGrid>& bilateral_grid,
                const param::StrategyParameters& opt_params);

            std::expected<torch::Tensor, std::string> compute_sparsity_loss(
                int iter,
                const SplatData& splatData);

            // Sparsity handling
            std::expected<void, std::string> handle_sparsity_update(
                int iter,
                SplatData& splatData);

            std::expected<void, std::string> apply_sparsity_pruning(
                int iter,
                SplatData& splatData);

            // Main training step
            std::expected<StepResult, std::string> train_step(
                int iter,
                Camera* cam,
                torch::Tensor gt_image,
                RenderMode render_mode,
                std::stop_token stop_token);

            // Member variables - core components
            std::shared_ptr<CameraDataset> base_dataset_;
            std::shared_ptr<CameraDataset> train_dataset_;
            std::shared_ptr<CameraDataset> val_dataset_;
            std::unique_ptr<IStrategy> strategy_;
            param::TrainingParameters params_;
            std::unordered_map<int, std::shared_ptr<Camera>> m_cam_id_to_cam;

            // Training components
            std::unique_ptr<BilateralGrid> bilateral_grid_;
            std::unique_ptr<torch::optim::Adam> bilateral_grid_optimizer_;
            std::unique_ptr<WarmupExponentialLR> bilateral_grid_scheduler_;
            std::unique_ptr<PoseOptimizationModule> poseopt_module_;
            std::unique_ptr<torch::optim::Adam> poseopt_optimizer_;
            std::unique_ptr<ISparsityOptimizer> sparsity_optimizer_;
            std::unique_ptr<MetricsEvaluator> evaluator_;
            std::unique_ptr<TrainingProgress> progress_;
            torch::Tensor background_;
            size_t train_dataset_size_ = 0;

            // State management
            std::atomic<bool> pause_requested_{false};
            std::atomic<bool> save_requested_{false};
            std::atomic<bool> stop_requested_{false};
            std::atomic<bool> is_paused_{false};
            std::atomic<bool> is_running_{false};
            std::atomic<bool> training_complete_{false};
            std::atomic<bool> ready_to_start_{false};
            std::atomic<bool> initialized_{false};
            std::atomic<int> current_iteration_{0};
            std::atomic<float> current_loss_{0.0f};

            // Synchronization
            mutable std::shared_mutex render_mutex_;
            std::mutex init_mutex_;

            // Async callback
            std::function<void()> callback_;
            at::cuda::CUDAStream callback_stream_ = at::cuda::getStreamFromPool(false);
            std::atomic<bool> callback_busy_{false};

            // Project reference
            std::shared_ptr<gs::management::Project> lf_project_;
        };
    } // namespace training
} // namespace gs