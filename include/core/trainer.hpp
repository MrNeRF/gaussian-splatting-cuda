#pragma once

#include "core/dataset.hpp"
#include "core/istrategy.hpp"
#include "core/parameters.hpp"
#include "core/training_progress.hpp"
#include <memory>
#include <torch/torch.h>

namespace gs {

    class Trainer {
    public:
        // Constructor that takes ownership of strategy and shares dataset
        Trainer(std::shared_ptr<CameraDataset> dataset,
                std::unique_ptr<IStrategy> strategy,
                const param::ModelParameters& model_params,
                const param::OptimizationParameters& optim_params);

        // Delete copy operations
        Trainer(const Trainer&) = delete;
        Trainer& operator=(const Trainer&) = delete;

        // Allow move operations
        Trainer(Trainer&&) = default;
        Trainer& operator=(Trainer&&) = default;

        // Main training method
        virtual void train();

        // Get the strategy (for external access if needed)
        IStrategy& get_strategy() { return *strategy_; }
        const IStrategy& get_strategy() const { return *strategy_; }

    protected:
        // Helper to create fresh dataloaders
        auto make_dataloader(int workers = 4) const;

        // Member variables
        std::shared_ptr<CameraDataset> dataset_;
        std::unique_ptr<IStrategy> strategy_;
        param::ModelParameters model_params_;
        param::OptimizationParameters optim_params_;

        torch::Tensor background_;
        std::unique_ptr<TrainingProgress> progress_;
        size_t dataset_size_;
    };

} // namespace gs