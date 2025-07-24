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

        // Viewer mode (all non-headless cases)
        std::println("Starting viewer mode...");

        // Determine initial title
        std::string title = "3DGS Viewer";
        if (!params.ply_path.empty()) {
            title = std::format("3DGS Viewer - {}", params.ply_path.filename().string());
        } else if (!params.dataset.data_path.empty()) {
            title = "3DGS Viewer - Training";
        } else {
            title = "3DGS Viewer - No Data";
        }

        auto viewer = std::make_unique<GSViewer>(title, 1280, 720);
        viewer->setAntiAliasing(params.optimization.antialiasing);
        viewer->setStoredParams(params);

        // Handle different startup scenarios
        if (params.viewer_mode && !params.ply_path.empty()) {
            // PLY viewing mode
            std::println("Loading PLY file: {}", params.ply_path.string());
            auto splat_result = load_ply(params.ply_path);
            if (!splat_result) {
                std::println(stderr, "Error loading PLY: {}", splat_result.error());
                std::println("Continuing with empty viewer...");
            } else {
                auto splat_data = std::make_unique<SplatData>(std::move(*splat_result));
                std::println("Loaded {} Gaussians", splat_data->size());
                viewer->setStandaloneModel(std::move(splat_data));
            }
        } else if (!params.dataset.data_path.empty()) {
            // Training mode with viewer
            std::println("Setting up training with visualization...");

            // Save config
            auto save_result = param::save_training_parameters_to_json(params, params.dataset.output_path);
            if (!save_result) {
                std::println(stderr, "Error saving config: {}", save_result.error());
                std::println("Continuing with empty viewer...");
            } else {
                auto setup_result = setupTraining(params);
                if (!setup_result) {
                    std::println(stderr, "Error: {}", setup_result.error());
                    std::println("Continuing with empty viewer...");
                } else {
                    // Get viewer from trainer
                    auto viewer_result = setup_result->trainer->create_and_get_viewer();
                    if (!viewer_result) {
                        std::println(stderr, "Error creating viewer: {}", viewer_result.error());
                        return -1;
                    }

                    // Use the trainer's viewer instead
                    auto trainer_viewer = *viewer_result;
                    trainer_viewer->setStoredParams(params);

                    // Start training in separate thread
                    std::jthread training_thread([&setup_result](std::stop_token stop_token) {
                        auto train_result = setup_result->trainer->train(stop_token);
                        if (!train_result) {
                            std::println(stderr, "Training error: {}", train_result.error());
                        }
                    });

                    // Run GUI in main thread
                    trainer_viewer->run();

                    // Request stop and wait
                    if (setup_result->trainer->is_running()) {
                        std::println("Requesting training stop...");
                        training_thread.request_stop();
                    }

                    return 0;
                }
            }
        } else {
            // Empty viewer mode
            std::println("Starting empty viewer (no data provided)");
            std::println("Use File menu to load PLY files or datasets");
        }

        // Run the viewer
        viewer->run();

        std::println("Viewer closed.");
        return 0;
    }

} // namespace gs