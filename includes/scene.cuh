#pragma once

#include <memory>
#include <vector>

class CameraInfo;
class GaussianModel;
struct ModelParameters;
struct SceneInfo;

class Scene {
public:
    Scene(GaussianModel& gaussians, const ModelParameters& params);
    [[nodiscard]] const std::vector<CameraInfo>& GetTraingingCameras() const;

private:
    GaussianModel& _gaussians;
    const ModelParameters& _params;

    std::unique_ptr<SceneInfo> _scene_infos;
};