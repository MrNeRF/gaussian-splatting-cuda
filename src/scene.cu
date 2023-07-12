#include "read_utils.cuh"
#include "scene.cuh"
#include <chrono>
#include <random>

Scene::Scene(const std::filesystem::path& path) : _path(path) {

    _scene_info = read_colmap_scene_info(path);
    // Use current time as seed for random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::shuffle(_scene_info->_cameras.begin(), _scene_info->_cameras.end(), generator);
}