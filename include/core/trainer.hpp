#pragma once

#include "core/dataset.hpp"
#include "core/istrategy.hpp"
#include "core/metrics.hpp"
#include "core/parameters.hpp"
#include "core/training_progress.hpp"
#include <memory>
#include <torch/torch.h>

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

        // Main training method
        virtual void train();

        // Evaluation method
        metrics::EvalMetrics evaluate(int iteration);

        // Get the strategy (for external access if needed)
        IStrategy& get_strategy() { return *strategy_; }
        const IStrategy& get_strategy() const { return *strategy_; }

    protected:
        // Helper to create fresh dataloaders
        auto make_train_dataloader(int workers = 4) const;
        auto make_val_dataloader(int workers = 1) const;

        // Member variables
        std::shared_ptr<CameraDataset> train_dataset_;
        std::shared_ptr<CameraDataset> val_dataset_;
        std::unique_ptr<IStrategy> strategy_;
        param::TrainingParameters params_;

        torch::Tensor background_;
        std::unique_ptr<TrainingProgress> progress_;
        size_t train_dataset_size_;
        size_t val_dataset_size_;

        // Metrics
        std::unique_ptr<metrics::PSNR> psnr_metric_;
        std::unique_ptr<metrics::SSIM> ssim_metric_;
        std::unique_ptr<metrics::LPIPS> lpips_metric_;
        std::unique_ptr<metrics::MetricsReporter> metrics_reporter_;

        void save_depth_visualization(const torch::Tensor& depth, int iteration, const std::string& prefix);
        torch::Tensor apply_depth_colormap(const torch::Tensor& depth_normalized);
    };

} // namespace gs