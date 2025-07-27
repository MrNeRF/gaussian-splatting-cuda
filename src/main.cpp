#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/training_setup.hpp"
#include <print>

int main(int argc, char* argv[]) {
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
