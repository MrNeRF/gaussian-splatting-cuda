#include "gaussian.cuh"
#include "parameters.cuh"
#include "read_utils.cuh"
#include "scene.cuh"
#include "scene_info.cuh"
#include <chrono>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

Scene::Scene(GaussianModel& gaussians, const ModelParameters& params) : _gaussians(gaussians),
                                                                        _params(params) {
    // Right now there is only support for colmap
    if (std::filesystem::exists(_params.source_path)) {
        _scene_infos = read_colmap_scene_info(_params.source_path);
    } else {
        std::cout << "Error: " << _params.source_path << " does not exist!" << std::endl;
        exit(-1);
    }
    // Use current time as seed for random generator
    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    // std::default_random_engine generator(seed);

    // std::shuffle(_gaussians._cameras.begin(), _gaussians._cameras.end(), generator);
}