#include "core/training_setup.hpp"
#include "core/mcmc.hpp"
#include "core/transforms_reader.hpp" // For generate_random_point_cloud
#include "loader/loader_service.hpp"
#include <format>
#include <print>

namespace gs {

    std::expected<TrainingSetup, std::string> setupTraining(const param::TrainingParameters& params) {
        // 1. Create loader service
        loader::LoaderService loader_service;

        // 2. Set up load options
        loader::LoadOptions load_options{
            .resolution = params.dataset.resolution,
            .images_folder = params.dataset.images,
            .validate_only = false,
            .progress = [](float percentage, const std::string& message) {
                std::println("[{:5.1f}%] {}", percentage, message);
            }};

        // 3. Load the dataset
        auto load_result = loader_service.load(params.dataset.data_path, load_options);
        if (!load_result) {
            return std::unexpected(std::format("Failed to load dataset: {}", load_result.error()));
        }

        std::println("Dataset loaded successfully using {} loader", load_result->loader_used);

        // 4. Handle the loaded data based on type
        return std::visit([&params, &load_result](auto&& data) -> std::expected<TrainingSetup, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<gs::SplatData>>) {
                // Direct PLY load - not supported for training
                return std::unexpected("Direct PLY loading is not supported for training. Please use a dataset format (COLMAP or Blender).");

            } else if constexpr (std::is_same_v<T, loader::SceneData>) {
                // Full scene data - set up training

                // Get point cloud or generate random one
                PointCloud point_cloud_to_use;
                if (data.point_cloud && data.point_cloud->size() > 0) {
                    point_cloud_to_use = *data.point_cloud;
                    std::println("Using point cloud with {} points", point_cloud_to_use.size());
                } else {
                    // Generate random point cloud if needed
                    std::println("No point cloud provided, using random initialization");
                    point_cloud_to_use = generate_random_point_cloud();
                }

                // Initialize model directly with point cloud
                auto splat_result = SplatData::init_model_from_pointcloud(
                    params,
                    load_result->scene_center,
                    point_cloud_to_use);

                if (!splat_result) {
                    return std::unexpected(std::format("Failed to initialize model: {}", splat_result.error()));
                }

                // Create strategy
                auto strategy = std::make_unique<MCMC>(std::move(*splat_result));

                // Create trainer
                auto trainer = std::make_unique<Trainer>(
                    data.cameras,
                    std::move(strategy),
                    params);

                return TrainingSetup{
                    .trainer = std::move(trainer),
                    .dataset = data.cameras,
                    .scene_center = load_result->scene_center};
            }
        },
                          load_result->data);
    }

} // namespace gs
