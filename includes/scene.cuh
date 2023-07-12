#pragma once

#include "gaussian.cuh"
#include "scene_info.cuh"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

class Scene {
public:
    Scene(const std::filesystem::path& path);

private:
    std::filesystem::path _path;
    GaussianModel _gaussians;
    std::unique_ptr<SceneInfo> _scene_info;
};