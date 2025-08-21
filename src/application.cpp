#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/training_setup.hpp"
#include "project/project.hpp"
#include "visualizer/visualizer.hpp"
#include <print>

namespace gs {

    int run_headless_app(std::unique_ptr<param::TrainingParameters> params) {
        if (params->dataset.data_path.empty()) {
            std::println(stderr, "Error: Headless mode requires --data-path");
            return -1;
        }

        std::println("Starting headless training...");

        auto project = gs::management::CreateNewProject(params->dataset, params->optimization);
        if (!project) {
            std::println(stderr, "project creation failed");
            return -1;
        }

        auto setup_result = gs::setupTraining(*params);
        if (!setup_result) {
            std::println(stderr, "Error: {}", setup_result.error());
            return -1;
        }

        setup_result->trainer->setProject(project);
        auto train_result = setup_result->trainer->train();
        if (!train_result) {
            std::println(stderr, "Training error: {}", train_result.error());
            return -1;
        }

        return 0;
    }

    int run_gui_app(std::unique_ptr<param::TrainingParameters> params) {

        // gui app
        std::println("Starting viewer mode...");

        // Create visualizer with options
        auto viewer = visualizer::Visualizer::create({.title = "LichtFeld Studio",
                                                      .width = 1280,
                                                      .height = 720,
                                                      .antialiasing = params->optimization.antialiasing,
                                                      .enable_cuda_interop = true});

        if (!params->dataset.project_path.empty() &&
            !std::filesystem::exists(params->dataset.project_path)) {
            std::println(stderr, "project file does not exists {}", params->dataset.project_path.string());
            return -1;
        }

        if (std::filesystem::exists(params->dataset.project_path)) {
            bool success = viewer->openProject(params->dataset.project_path);
            if (!success) {
                std::println(stderr, "error opening existing project");
                return -1;
            }
            if (!params->ply_path.empty()) {
                std::println(stderr, "can not open ply and open project from commandline");
                return -1;
            }
            if (!params->dataset.data_path.empty()) {
                std::println(stderr, "cannot open new data_path and project from commandline");
                return -1;
            }
        }

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

    int Application::run(std::unique_ptr<param::TrainingParameters> params) {
        // no gui
        if (params->optimization.headless) {
            return run_headless_app(std::move(params));
        }
        // gui app
        return run_gui_app(std::move(params));
    }
} // namespace gs
