#include "core/training_setup.hpp"
#include "core/dataset_reader.hpp"
#include "core/mcmc.hpp"
#include <format>

namespace gs {

    std::expected<TrainingSetup, std::string> setupTraining(const param::TrainingParameters& params) {
        // 1. Get valid data reader
        auto dataSetReader = GetValidDataReader(params.dataset);

        // 2. Create dataset
        auto dataset_result = dataSetReader->create_dataset();
        if (!dataset_result) {
            return std::unexpected(std::format("Error creating dataset: {}", dataset_result.error()));
        }
        auto [dataset, scene_center] = std::move(*dataset_result);

        // 3. Initialize model
        auto splat_result = SplatData::init_model_from_pointcloud(params, scene_center, std::move(dataSetReader));
        if (!splat_result) {
            return std::unexpected(std::format("Error initializing model: {}", splat_result.error()));
        }

        // 4. Create strategy
        auto strategy = std::make_unique<MCMC>(std::move(*splat_result));

        // 5. Create trainer
        auto trainer = std::make_unique<Trainer>(
            dataset,
            std::move(strategy),
            params);

        return TrainingSetup{
            .trainer = std::move(trainer),
            .dataset = std::move(dataset),
            .scene_center = scene_center};
    }

} // namespace gs