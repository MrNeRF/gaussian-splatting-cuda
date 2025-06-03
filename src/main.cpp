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
        auto params = gs::args::parse_args_and_params(argc, argv);
        //----------------------------------------------------------------------
        // 2. Create dataset from COLMAP
        //----------------------------------------------------------------------
        auto [dataset, scene_scale] = create_dataset_from_colmap(params.dataset);
        //----------------------------------------------------------------------
        // 3. Model initialisation
        //----------------------------------------------------------------------
        auto splat_data = SplatData::init_model_from_pointcloud(params, scene_scale);
        //----------------------------------------------------------------------
        // 4. Create strategy
        //----------------------------------------------------------------------
        auto strategy = std::make_unique<MCMC>(std::move(splat_data));
        //----------------------------------------------------------------------
        // 5. Create trainer
        //----------------------------------------------------------------------
        gs::Trainer trainer(dataset, std::move(strategy), params);
        //----------------------------------------------------------------------
        // 6. Start training
        //----------------------------------------------------------------------
        trainer.train();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}