#pragma once

#include "scene_info.cuh"
#include <memory>
#include <vector>

class CameraInfo;
class GaussianModel;
class Camera;
namespace gs::param {
    struct ModelParameters;
}

class TrainingData {
public:
    TrainingData(const gs::param::ModelParameters& params);
    void Init_model(GaussianModel& gaussians);
    Camera& Get_training_camera();
    [[nodiscard]] float Get_cameras_extent() const { return static_cast<float>(_scene_infos->_nerf_norm_radius); }

private:
    const gs::param::ModelParameters& _params;
    std::vector<Camera> _cameras;
    std::unique_ptr<SceneInfo> _scene_infos;
    std::vector<int> _indices;
};