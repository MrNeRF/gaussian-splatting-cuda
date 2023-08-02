#include "gaussian.cuh"
#include "parameters.cuh"
#include "read_utils.cuh"
#include "scene.cuh"
#include "scene_info.cuh"
#include <chrono>
#include <filesystem>
#include <random>
#include <string>

// TODO: support start from later iterations. Compare original code
// We also have only training, no testing
// TODO: support also testing
Scene::Scene(GaussianModel& gaussians, const ModelParameters& params) : _gaussians(gaussians),
                                                                        _params(params) {
    // Right now there is only support for colmap
    if (std::filesystem::exists(_params.source_path)) {
        _scene_infos = read_colmap_scene_info(_params.source_path);
    } else {
        std::cout << "Error: " << _params.source_path << " does not exist!" << std::endl;
        exit(-1);
    }

    // TODO: json camera dumping for debugging purpose at least
    // Use current time as seed for random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    // std::shuffle(_scene_infos->_cameras.begin(), _scene_infos->_cameras.end(), generator);

    // get the parameterr self.cameras.extent
    _gaussians.Create_from_pcd(_scene_infos->_point_cloud, _scene_infos->_nerf_norm_radius);
}
const std::vector<CameraInfo>& Scene::GetTraingingCameras() const {
    return _scene_infos->_cameras;
}
