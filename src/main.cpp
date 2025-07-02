#include "core/argument_parser.hpp"
#include "core/dataset.hpp"
// #include "core/knn_utils.hpp" // Now included via setup_utils.hpp if needed there
#include "core/setup_utils.hpp" // Added for KNN setup utility
#include "core/mcmc.hpp"
#include "core/parameters.hpp"
#include "core/trainer.hpp"
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
        // Now also returns camera_world_positions
        auto [dataset, scene_center, camera_world_positions] = create_dataset_from_colmap(params.dataset);

        //----------------------------------------------------------------------
        // 4. Model initialisation
        //----------------------------------------------------------------------
        auto splat_data_obj = SplatData::init_model_from_pointcloud(params, scene_center);

        //----------------------------------------------------------------------
        // 4.1 Setup Camera KNN data using the utility function
        //----------------------------------------------------------------------
        gs::utils::setup_camera_knn_for_splat_data(
            splat_data_obj,
            dataset,
            camera_world_positions,
            scene_center,
            params.optimization
        );

        //----------------------------------------------------------------------
        // 5. Create strategy
        //----------------------------------------------------------------------
        // splat_data_obj is moved into MCMC strategy here
        auto strategy = std::make_unique<MCMC>(std::move(splat_data_obj));

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