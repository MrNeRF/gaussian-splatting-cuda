#pragma once

#include "scene_info.cuh"
#include <memory>

class GaussianModel;
struct ModelParameters;

class Scene {
public:
    Scene(GaussianModel& gaussians, const ModelParameters& params);

private:
    GaussianModel& _gaussians;
    const ModelParameters& _params;

    std::unique_ptr<SceneInfo> _scene_info;
};