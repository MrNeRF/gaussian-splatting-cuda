#include "core/argument_parser.hpp"
#include "core/dataset.hpp"
#include "core/inria_adc.hpp"
#include "core/parameters.hpp"
#include "core/trainer.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    //----------------------------------------------------------------------
    // 1. Parse command-line arguments
    //----------------------------------------------------------------------
    auto args = gs::args::convert_args(argc, argv);
    auto modelParams = gs::param::ModelParameters();
    auto optimParams = gs::param::read_optim_params_from_json();
    if (gs::args::parse_arguments(args, modelParams, optimParams) < 0)
        return -1;

    //----------------------------------------------------------------------
    // 2. Create dataset
    //----------------------------------------------------------------------
    auto dataset = create_camera_dataset(modelParams);
    auto scene = dataset->get_scene_info();

    //----------------------------------------------------------------------
    // 3. Model initialisation
    //----------------------------------------------------------------------
    auto splat_data = SplatData::create_from_point_cloud(scene._point_cloud,
                                                         modelParams.sh_degree,
                                                         scene._nerf_norm_radius);

    auto strategy = std::make_unique<InriaADC>(modelParams.sh_degree, std::move(splat_data));

    //----------------------------------------------------------------------
    // 4. Create trainer and run training
    //----------------------------------------------------------------------
    gs::Trainer trainer(dataset, std::move(strategy), modelParams, optimParams);
    trainer.train();

    return 0;
}