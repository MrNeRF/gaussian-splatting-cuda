#include "camera.cuh"
#include "camera_utils.cuh"
#include "gaussian.cuh"
#include "parameters.cuh"
#include "read_utils.cuh"
#include "scene.cuh"

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

    _cameras.reserve(_scene_infos->_cameras.size());
    std::vector<nlohmann::json> json_cams;
    json_cams.reserve(_scene_infos->_cameras.size());
    int counter = 0;
    for (auto& cam_info : _scene_infos->_cameras) {
        _cameras.emplace_back(loadCam(_params, counter, cam_info));
        json_cams.push_back(Convert_camera_to_JSON(cam_info, counter, _cameras.back().Get_R(), _cameras.back().Get_T()));
        ++counter;
    }
    dump_JSON(params.output_path / "cameras.json", json_cams);
    // TODO: json camera dumping for debugging purpose at least

    // get the parameterr self.cameras.extent
    _gaussians.Create_from_pcd(_scene_infos->_point_cloud, _scene_infos->_nerf_norm_radius);
}