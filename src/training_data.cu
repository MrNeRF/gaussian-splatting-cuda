#include "camera.cuh"
#include "gaussian.cuh"
#include "parameters.cuh"
#include <random>
#include "read_utils.cuh"
#include "training_data.cuh"

std::vector<int> get_random_indices(int max_index) {
    std::vector<int> indices(max_index);
    std::iota(indices.begin(), indices.end(), 0);
    // Shuffle the vector
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
    std::reverse(indices.begin(), indices.end());
    return indices;
}

TrainingData::TrainingData(const gs::param::ModelParameters& params): _params(params) {
    // Right now there is only support for colmap
    if (std::filesystem::exists(_params.source_path)) {
        _scene_infos = read_colmap_scene_info(_params.source_path, _params.resolution);
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
}

void TrainingData::Init_model(GaussianModel& gaussians) {
    gaussians.Create_from_pcd(_scene_infos->_point_cloud, _scene_infos->_nerf_norm_radius);
}

Camera& TrainingData::Get_training_camera() {
    if (_indices.empty()) {
        _indices = get_random_indices(_cameras.size());
    }

    const int camera_index = _indices.back();
    _indices.pop_back(); // remove last element to iterate over all cameras randomly
    return _cameras[camera_index];
}
