/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "dataset.hpp"
#include "trainer.hpp"
#include <expected>
#include <memory>

namespace gs::training {
    struct TrainingSetup {
        std::unique_ptr<Trainer> trainer;
        std::shared_ptr<CameraDataset> dataset;
        torch::Tensor scene_center;
    };

    // Reusable function to set up training from parameters
    std::expected<TrainingSetup, std::string> setupTraining(const param::TrainingParameters& params);
} // namespace gs::training
