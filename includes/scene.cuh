#pragma once

#include <memory>

class GaussianModel;
struct ModelParameters;
struct SceneInfo;

class Scene {
public:
    Scene(GaussianModel& gaussians, const ModelParameters& params);

private:
    GaussianModel& _gaussians;
    const ModelParameters& _params;

    std::unique_ptr<SceneInfo> _scene_infos;
};