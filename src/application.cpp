#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "visualizer/visualizer.hpp"
#include <print>

namespace gs {
    int Application::run(std::unique_ptr<param::TrainingParameters> params) {
        // Create visualizer with options
        auto viewer = visualizer::Visualizer::create({.title = "LichtFeld Studio",
                                                      .width = 1280,
                                                      .height = 720,
                                                      .antialiasing = params->optimization.antialiasing,
                                                      .enable_cuda_interop = true});

        // Set parameters
        viewer->setParameters(*params);

        // Load data if specified
        if (!params->ply_path.empty()) {
            auto result = viewer->loadPLY(params->ply_path);
            if (!result) {
                std::println(stderr, "Error: {}", result.error());
                return -1;
            }
        } else if (!params->dataset.data_path.empty()) {
            auto result = viewer->loadDataset(params->dataset.data_path);
            if (!result) {
                std::println(stderr, "Error: {}", result.error());
                return -1;
            }
        }

        std::println("Anti-aliasing: {}", params->optimization.antialiasing ? "enabled" : "disabled");

        // Run the viewer
        viewer->run();

        std::println("Viewer closed.");
        return 0;
    }
} // namespace gs
