#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/ply_loader.hpp"
#include "core/training_setup.hpp"
#include "visualizer/detail.hpp"
#include <print>
#include <thread>

namespace gs {

    int Application::run(int argc, char* argv[]) {
        // Parse arguments
        auto params_result = args::parse_args_and_params(argc, argv);
        if (!params_result) {
            std::println(stderr, "Error: {}", params_result.error());
            return -1;
        }
        auto params = std::move(*params_result);

        if (params.viewer_mode) {
            // PLY Viewer Mode
            std::println("Loading PLY file: {}", params.ply_path.string());

            auto splat_result = load_ply(params.ply_path);
            if (!splat_result) {
                std::println(stderr, "Error loading PLY: {}", splat_result.error());
                return -1;
            }

            auto splat_data = std::make_unique<SplatData>(std::move(*splat_result));
            std::println("Loaded {} Gaussians", splat_data->size());

            // Create viewer without trainer
            std::string title = "3DGS Viewer - " + params.ply_path.filename().string();
            auto viewer = std::make_unique<GSViewer>(title, 1280, 720);
            viewer->setTrainer(nullptr); // No trainer in viewer mode
            viewer->setStandaloneModel(std::move(splat_data));
            viewer->setAntiAliasing(params.optimization.antialiasing);

            std::println("Starting viewer...");
            std::println("Anti-aliasing: {}", params.optimization.antialiasing ? "enabled" : "disabled");

            viewer->run();

            std::println("Viewer closed.");
            return 0;
        }
        // Headless mode - requires data path
        if (params.optimization.headless) {
            if (params.dataset.data_path.empty()) {
                std::println(stderr, "Error: Headless mode requires --data-path");
                return -1;
            }

            std::println("Starting headless training...");

            // Save config
            auto save_result = param::save_training_parameters_to_json(params, params.dataset.output_path);
            if (!save_result) {
                std::println(stderr, "Error saving config: {}", save_result.error());
                return -1;
            }

            auto setup_result = setupTraining(params);
            if (!setup_result) {
                std::println(stderr, "Error: {}", setup_result.error());
                return -1;
            }

            auto train_result = setup_result->trainer->train();
            if (!train_result) {
                std::println(stderr, "Training error: {}", train_result.error());
                return -1;
            }

            return 0;
        }

        auto setup_result = setupTraining(params);
        std::println("Starting viewer mode...");
        auto viewer_result = setup_result->trainer->create_and_get_viewer();
        if (!viewer_result) {
            std::println(stderr, "Error creating viewer: {}", viewer_result.error());
            return -1;
        }
        auto viewer = *viewer_result;

        // Start training in a separate jthread with stop token support
        std::jthread training_thread([trainer_ptr = setup_result->trainer.get()](std::stop_token stop_token) {
            auto train_result = trainer_ptr->train(stop_token);
            if (!train_result) {
                std::println(stderr, "Training error: {}", train_result.error());
            }
        });

        // Run GUI in main thread (blocking)
        viewer->run();

        // Request cancellation when GUI closes
        if (setup_result->trainer->is_running()) {
            std::println("Main: Requesting training stop...");
            training_thread.request_stop();
        }

        // jthread automatically joins when destroyed
        std::println("Main: Waiting for training thread to finish...");
        // training_thread destructor will join automatically
        std::println("Main: Training thread finished.");

        return 0;
    }

} // namespace gs