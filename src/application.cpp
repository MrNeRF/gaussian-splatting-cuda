#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/ply_loader.hpp"
#include "core/training_setup.hpp"
#include "visualizer/detail.hpp"
#include <print>

namespace gs {

    int Application::run(std::unique_ptr<param::TrainingParameters> params) {
        auto viewer = std::make_unique<GSViewer>("3DGS Viewer", 1280, 720);
        viewer->setAntiAliasing(params->optimization.antialiasing);

        // If a PLY was specified via command line, load it
        if (!params->ply_path.empty()) {
            viewer->loadPLYFile(params->ply_path);
        }
        // If a dataset was specified via command line, load it
        else if (!params->dataset.data_path.empty()) {
            viewer->loadDataset(params->dataset.data_path);
        }
        // Otherwise start with empty viewer

        std::println("Anti-aliasing: {}", params->optimization.antialiasing ? "enabled" : "disabled");

        viewer->run();

        std::println("Viewer closed.");
        return 0;
    }

} // namespace gs