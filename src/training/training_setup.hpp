#pragma once

#include "dataset.hpp"
#include "trainer.hpp"
#include <expected>
#include <memory>

namespace gs {

    struct TrainingSetup {
        std::unique_ptr<Trainer> trainer;
        std::shared_ptr<CameraDataset> dataset;
        torch::Tensor scene_center;
    };

    // Reusable function to set up training from parameters
    std::expected<TrainingSetup, std::string> setupTraining(const param::TrainingParameters& params);

} // namespace gs