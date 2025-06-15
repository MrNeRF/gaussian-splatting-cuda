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
        void train();

    private:
        // Protected method for processing a single training step
        // Returns true if training should continue
        bool train_step(int iter, Camera* cam, torch::Tensor gt_image, RenderMode render_mode);

        // Protected method for computing loss
        torch::Tensor compute_loss(const RenderOutput& render_output,
                                   const torch::Tensor& gt_image,
                                   const SplatData& splatData,
                                   const param::OptimizationParameters& opt_params);

        // Member variables
        std::shared_ptr<CameraDataset> train_dataset_;
        std::shared_ptr<CameraDataset> val_dataset_;
        std::unique_ptr<IStrategy> strategy_;
        param::TrainingParameters params_;

        torch::Tensor background_;
        std::unique_ptr<TrainingProgress> progress_;
        size_t train_dataset_size_;

        // Metrics evaluator - handles all evaluation logic
        std::unique_ptr<metrics::MetricsEvaluator> evaluator_;
    };

} // namespace gs