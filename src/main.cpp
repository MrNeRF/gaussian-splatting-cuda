#include "core/argument_parser.hpp"
#include "core/dataset.hpp"
#include "core/mcmc.hpp"
#include "core/parameters.hpp"
#include "core/trainer.hpp"
#include <iostream>

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
        auto [dataset, scene_center] = create_dataset_from_colmap(params.dataset);

        //----------------------------------------------------------------------
        // 4. Model initialisation
        //----------------------------------------------------------------------
        auto splat_data = SplatData::init_model_from_pointcloud(params, scene_center);

        //----------------------------------------------------------------------
        // 5. Create strategy
        //----------------------------------------------------------------------
        auto strategy = std::make_unique<MCMC>(std::move(splat_data));

        //----------------------------------------------------------------------
        // 6. Create trainer
        //----------------------------------------------------------------------
        gs::Trainer trainer(dataset, std::move(strategy), params);

        //----------------------------------------------------------------------
        // 7. Start training
        //----------------------------------------------------------------------
        trainer.train();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}