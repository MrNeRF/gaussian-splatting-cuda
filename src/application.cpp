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

        // Viewer mode - create empty viewer
        std::println("Starting viewer mode...");

        auto viewer = std::make_unique<GSViewer>("3DGS Viewer", 1280, 720);
        viewer->setAntiAliasing(params.optimization.antialiasing);

        // If a PLY was specified via command line, load it
        if (!params.ply_path.empty()) {
            viewer->loadPLYFile(params.ply_path);
        }
        // If a dataset was specified via command line, load it
        else if (!params.dataset.data_path.empty()) {
            viewer->loadDataset(params.dataset.data_path);
        }
        // Otherwise start with empty viewer

        std::println("Anti-aliasing: {}", params.optimization.antialiasing ? "enabled" : "disabled");

        viewer->run();

        std::println("Viewer closed.");
        return 0;
    }

} // namespace gs