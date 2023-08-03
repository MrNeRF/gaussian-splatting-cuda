#pragma once

#include <memory>
#include <vector>

class CameraInfo;
class GaussianModel;
struct ModelParameters;
struct SceneInfo;
class Camera;

class Scene {
public:
    Scene(GaussianModel& gaussians, const ModelParameters& params);
    int Get_camera_count() const { return _scene_infos->_cameras.size(); }
    Camera& Get_training_camera(int i) { return _cameras[i]; }

private:
    GaussianModel& _gaussians;
    const ModelParameters& _params;
    std::vector<Camera> _cameras;
    std::unique_ptr<SceneInfo> _scene_infos;
};