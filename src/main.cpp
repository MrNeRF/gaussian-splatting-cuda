#include "core/argument_parser.hpp"
#include "core/dataset.hpp"
#include "core/mcmc.hpp"
#include "core/parameters.hpp"
#include "core/trainer.hpp"
#include "core/newton_strategy.hpp" // Added for NewtonStrategy
#include "core/setup_utils.hpp"
#include "visualizer/detail.hpp"
#include <iostream>
#include <memory>
#include <thread>

int main(int argc, char* argv[]) {
    try {
        //----------------------------------------------------------------------
        // 1. Parse arguments and load parameters in one step
        //----------------------------------------------------------------------
        const auto params = gs::args::parse_args_and_params(argc, argv);

        //----------------------------------------------------------------------
        // 2. Save training configuration to output directory
        //----------------------------------------------------------------------
        gs::param::save_training_parameters_to_json(params, params.dataset.output_path);

        //----------------------------------------------------------------------
        // 3. Create dataset from COLMAP
        //----------------------------------------------------------------------
        auto [dataset, scene_center, camera_world_positions] = create_dataset_from_colmap(params.dataset);

        //----------------------------------------------------------------------
        // 4. Model initialisation
        //----------------------------------------------------------------------
        auto splat_data = SplatData::init_model_from_pointcloud(params, scene_center);

	gs::utils::setup_camera_knn_for_splat_data(splat_data,dataset,camera_world_positions,scene_center,params.optimization);

        //----------------------------------------------------------------------
        // 5. Create strategy
        //----------------------------------------------------------------------
        std::unique_ptr<IStrategy> strategy;
        // splat_data was initialized earlier and KNNs were set up in it.
        // Now, create a unique_ptr for it to pass to the strategy.
        auto splat_data_owner = std::make_unique<SplatData>(std::move(splat_data));

        if (params.optimization.use_newton_optimizer) {
            std::cout << "INFO: Using NewtonStrategy." << std::endl;
            // NewtonStrategy constructor: std::unique_ptr<SplatData> splat_data_owner, std::shared_ptr<CameraDataset> train_dataset_for_knn
            // Using direct construction instead of std::make_unique to potentially resolve C2665
            strategy = std::unique_ptr<NewtonStrategy>(new NewtonStrategy(std::move(splat_data_owner), dataset));
        }
        // else if (params.optimization.use_mcmc_strategy) { // Example if there was an explicit MCMC flag
        //     std::cout << "INFO: Using MCMCStrategy." << std::endl;
        //     strategy = std::make_unique<MCMC>(std::move(splat_data_owner));
        // }
        else {
            // Default strategy: MCMC as it was the previous default.
            // Or, could be an error, or a BasicAdamStrategy if one existed.
            std::cout << "INFO: use_newton_optimizer is false. Defaulting to MCMCStrategy." << std::endl;
            strategy = std::make_unique<MCMC>(std::move(splat_data_owner));
        }

        TORCH_CHECK(strategy, "Strategy could not be initialized!");

        //----------------------------------------------------------------------
        // 6. Create trainer
        //----------------------------------------------------------------------
        auto trainer = std::make_unique<gs::Trainer>(dataset, std::move(strategy), params);

        //----------------------------------------------------------------------
        // 7. Start training based on visualization mode
        //----------------------------------------------------------------------
        if (params.optimization.enable_viz) {
            // GUI Mode: Create viewer and run it in main thread
            auto viewer = trainer->create_and_get_viewer();
            if (viewer) {
                // Start training in a separate thread
                std::thread training_thread([&trainer]() {
                    try {
                        trainer->train();
                    } catch (const std::exception& e) {
                        std::cerr << "Training thread error: " << e.what() << std::endl;
                    }
                });

                // Run GUI in main thread (blocking)
                viewer->run();

                // After viewer closes, ensure training is stopped
                if (trainer->is_running()) {
                    std::cout << "Main: Requesting training stop..." << std::endl;
                    trainer->request_stop();
                }

                // Wait for training thread to complete
                if (training_thread.joinable()) {
                    std::cout << "Main: Waiting for training thread to finish..." << std::endl;
                    training_thread.join();
                    std::cout << "Main: Training thread finished." << std::endl;
                }
            } else {
                std::cerr << "Failed to create viewer" << std::endl;
                return -1;
            }
        } else {
            // Headless Mode: Run training in main thread
            trainer->train();
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
