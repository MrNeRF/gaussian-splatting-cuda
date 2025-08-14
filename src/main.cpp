#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/training_setup.hpp"
#include "management/project.hpp"
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <print>

int main(int argc, char* argv[]) {
//----------------------------------------------------------------------
// 0. Set CUDA caching allocator settings to avoid fragmentation issues
// This avoids the need to repeatedly call emptyCache() after
// densification steps. We manually call the proper function here
// instead of setting the environment variable hoping that this then
// also works on Windows. Setting the environment variable using
// setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True", 1);
// would work on Linux but not on Windows, so we use the C++ API.
// Should this break in the future, we can always revert to the old
// approach of calling emptyCache() after each densification step.
//----------------------------------------------------------------------
#ifndef _WIN32
    // Windows doesn't support CUDACachingAllocator expandable_segments
    c10::cuda::CUDACachingAllocator::setAllocatorSettings("expandable_segments:True");
#endif

    auto params_result = gs::args::parse_args_and_params(argc, argv);
    if (!params_result) {
        std::println(stderr, "Error: {}", params_result.error());
        return -1;
    }
    auto params = std::move(*params_result);
    // no gui
    if (params->optimization.headless) {

        if (params->dataset.data_path.empty()) {
            std::println(stderr, "Error: Headless mode requires --data-path");
            return -1;
        }

        std::println("Starting headless training...");

        auto project = std::make_shared<gs::management::LichtFeldProjectFile>();
        project->setProjectName("project");
        project->setOutputFileName( params->dataset.output_path / "lichtfeld.ls");

        // Save config
        auto save_result = gs::param::save_training_parameters_to_json(*params, params->dataset.output_path);
        if (!save_result) {
            std::println(stderr, "Error saving config: {}", save_result.error());
            return -1;
        }

        auto setup_result = gs::setupTraining(*params);
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

    // gui app
    std::println("Starting viewer mode...");
    gs::Application app;
    return app.run(std::move(params));
}
