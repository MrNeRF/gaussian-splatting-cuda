#include "gaussian.cuh"
#include "parameters.cuh"
#include "read_utils.cuh"
#include "scene.cuh"
#include <chrono>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

Scene::Scene(GaussianModel& gaussians, const ModelParameters& params) : _gaussians(gaussians),
                                                                        _params(params) {
    // Use current time as seed for random generator
    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    // std::default_random_engine generator(seed);

    // std::shuffle(_gaussians._cameras.begin(), _gaussians._cameras.end(), generator);
}