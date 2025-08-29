/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_setup.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "loader/loader.hpp"
#include "strategies/default_strategy.hpp"
#include "strategies/mcmc.hpp"
#include <format>

namespace gs::training {
    std::expected<TrainingSetup, std::string> setupTraining(const param::TrainingParameters& params) {
        // 1. Create loader
        auto loader = loader::Loader::create();

        // 2. Set up load options
        loader::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .images_folder = params.dataset.images,
            .validate_only = false,
            .progress = [](float percentage, const std::string& message) {
                LOG_DEBUG("[{:5.1f}%] {}", percentage, message);
            }};

        // 3. Load the dataset
        LOG_INFO("Loading dataset from: {}", params.dataset.data_path.string());
        auto load_result = loader->load(params.dataset.data_path, load_options);
        if (!load_result) {
            return std::unexpected(std::format("Failed to load dataset: {}", load_result.error()));
        }

        LOG_INFO("Dataset loaded successfully using {} loader", load_result->loader_used);

        // 4. Handle the loaded data based on type
        return std::visit([&params, &load_result](auto&& data) -> std::expected<TrainingSetup, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<gs::SplatData>>) {
                // Direct PLY load - not supported for training
                return std::unexpected(
                    "Direct PLY loading is not supported for training. Please use a dataset format (COLMAP or Blender).");
            } else if constexpr (std::is_same_v<T, loader::LoadedScene>) {
                // Full scene data - set up training

                // Get point cloud or generate random one
                PointCloud point_cloud_to_use;
                if (data.point_cloud && data.point_cloud->size() > 0) {
                    point_cloud_to_use = *data.point_cloud;
                    LOG_INFO("Using point cloud with {} points", point_cloud_to_use.size());
                } else {
                    // Generate random point cloud if needed
                    LOG_INFO("No point cloud provided, using random initialization");
                    // Need to generate random point cloud - this should be provided by the loader or a utility
                    int numInitGaussian = 10000;
                    uint64_t seed = 8128;
                    torch::manual_seed(seed);

                    torch::Tensor positions = torch::rand({numInitGaussian, 3}); // in [0, 1]
                    positions = positions * 2.0 - 1.0;                           // now in [-1, 1]
                    torch::Tensor colors =
                        torch::randint(0, 256, {numInitGaussian, 3}, torch::kUInt8);

                    point_cloud_to_use = PointCloud(positions, colors);
                }

                // Initialize model directly with point cloud
                auto splat_result = SplatData::init_model_from_pointcloud(
                    params,
                    load_result->scene_center,
                    point_cloud_to_use);

                if (!splat_result) {
                    return std::unexpected(
                        std::format("Failed to initialize model: {}", splat_result.error()));
                }

                // 5. Create strategy
                std::unique_ptr<IStrategy> strategy;
                if (params.optimization.strategy == "mcmc") {
                    strategy = std::make_unique<MCMC>(std::move(*splat_result));
                    LOG_DEBUG("Created MCMC strategy");
                } else {
                    strategy = std::make_unique<DefaultStrategy>(std::move(*splat_result));
                    LOG_DEBUG("Created default strategy");
                }

                // Create trainer (without parameters)
                auto trainer = std::make_unique<Trainer>(
                    data.cameras,
                    std::move(strategy));

                return TrainingSetup{
                    .trainer = std::move(trainer),
                    .dataset = data.cameras,
                    .scene_center = load_result->scene_center};
            } else {
                return std::unexpected("Unknown data type returned from loader");
            }
        },
                          load_result->data);
    }
} // namespace gs::training