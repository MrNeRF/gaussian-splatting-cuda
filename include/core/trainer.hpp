#pragma once

#include "core/bilateral_grid.hpp"
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
                const param::TrainingParameters& params);

        // Delete copy operations
        Trainer(const Trainer&) = delete;
        Trainer& operator=(const Trainer&) = delete;

        // Allow move operations
        Trainer(Trainer&&) = default;
        Trainer& operator=(Trainer&&) = default;

        // Main training method
        virtual void train();

    protected:
        // Helper methods
        auto make_dataloader(int workers = 4) const;
        void initialize_bilateral_grid();
        torch::Tensor ensure_4d(const torch::Tensor& image) const;
        torch::Tensor compute_losses(const torch::Tensor& rendered,
                                     const torch::Tensor& ground_truth);
        torch::Tensor compute_regularization_losses();
        void step_optimizers(int iter);
        void save_checkpoint(int iter);

        // Member variables
        std::shared_ptr<CameraDataset> dataset_;
        std::unique_ptr<IStrategy> strategy_;
        param::TrainingParameters params_;

        torch::Tensor background_;
        std::unique_ptr<TrainingProgress> progress_;
        size_t dataset_size_;

        // Bilateral grid components
        std::unique_ptr<gs::BilateralGrid> bilateral_grid_;
        std::unique_ptr<torch::optim::Adam> bilateral_grid_optimizer_;
    };

} // namespace gs