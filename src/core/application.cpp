/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/logger.hpp"
#include "project/project.hpp"
#include "training/training_setup.hpp"
#include "visualizer/visualizer.hpp"

namespace gs {

    int run_headless_app(std::unique_ptr<param::TrainingParameters> params) {
        if (params->dataset.data_path.empty()) {
            LOG_ERROR("Headless mode requires --data-path");
            return -1;
        }

        LOG_INFO("Starting headless training...");

        auto project = gs::management::CreateNewProject(params->dataset, params->optimization);
        if (!project) {
            LOG_ERROR("Project creation failed");
            return -1;
        }

        auto setup_result = gs::training::setupTraining(*params);
        if (!setup_result) {
            LOG_ERROR("Training setup failed: {}", setup_result.error());
            return -1;
        }

        setup_result->trainer->setProject(project);

        // Initialize trainer in headless mode with parameters
        auto init_result = setup_result->trainer->initialize(*params);
        if (!init_result) {
            LOG_ERROR("Failed to initialize trainer: {}", init_result.error());
            return -1;
        }

        auto train_result = setup_result->trainer->train();
        if (!train_result) {
            LOG_ERROR("Training error: {}", train_result.error());
            return -1;
        }

        LOG_INFO("Headless training completed successfully");
        return 0;
    }

    int run_gui_app(std::unique_ptr<param::TrainingParameters> params) {
        LOG_INFO("Starting viewer mode...");

        LOG_DEBUG("removing temporary projects");
        gs::management::RemoveTempUnlockedProjects();

        // Create visualizer with options
        auto viewer = visualizer::Visualizer::create({.title = "LichtFeld Studio",
                                                      .width = 1280,
                                                      .height = 720,
                                                      .antialiasing = params->optimization.antialiasing,
                                                      .enable_cuda_interop = true,
                                                      .gut = params->optimization.gut});

        if (!params->dataset.project_path.empty() &&
            !std::filesystem::exists(params->dataset.project_path)) {
            LOG_ERROR("Project file does not exist: {}", params->dataset.project_path.string());
            return -1;
        }

        if (std::filesystem::exists(params->dataset.project_path)) {
            bool success = viewer->openProject(params->dataset.project_path);
            if (!success) {
                LOG_ERROR("Error opening existing project");
                return -1;
            }
            if (!params->ply_path.empty()) {
                LOG_ERROR("Cannot open PLY and project from command line simultaneously");
                return -1;
            }
            if (!params->dataset.data_path.empty()) {
                LOG_ERROR("Cannot open new data_path and project from command line simultaneously");
                return -1;
            }
        } else { // create temporary project until user will save it in desired location
            std::shared_ptr<gs::management::Project> project = nullptr;
            if (params->dataset.output_path.empty()) {
                project = gs::management::CreateTempNewProject(params->dataset, params->optimization);
                if (!project) {
                    LOG_ERROR("Temporary project creation failed");
                    return -1;
                }
                params->dataset.output_path = project->getProjectOutputFolder();
                LOG_DEBUG("Created temporary project at: {}", params->dataset.output_path.string());
            } else {
                project = gs::management::CreateNewProject(params->dataset, params->optimization);
                if (!project) {
                    LOG_ERROR("Project creation failed");
                    return -1;
                }
                LOG_DEBUG("Created project at: {}", params->dataset.output_path.string());
            }
            viewer->attachProject(project);
        }

        // Set parameters
        viewer->setParameters(*params);

        // Load data if specified
        if (!params->ply_path.empty()) {
            LOG_INFO("Loading PLY file: {}", params->ply_path.string());
            auto result = viewer->loadPLY(params->ply_path);
            if (!result) {
                LOG_ERROR("Failed to load PLY: {}", result.error());
                return -1;
            }
        } else if (!params->dataset.data_path.empty()) {
            LOG_INFO("Loading dataset: {}", params->dataset.data_path.string());
            auto result = viewer->loadDataset(params->dataset.data_path);
            if (!result) {
                LOG_ERROR("Failed to load dataset: {}", result.error());
                return -1;
            }
        }

        LOG_INFO("Anti-aliasing: {}", params->optimization.antialiasing ? "enabled" : "disabled");

        // Run the viewer
        viewer->run();

        LOG_INFO("Viewer closed");
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