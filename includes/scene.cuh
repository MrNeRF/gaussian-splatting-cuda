#pragma once

#include "point_cloud.cuh"
#include <filesystem>
#include <string>
#include <vector>

struct SceneInfo {
};

class Scene {
public:
    Scene(const std::filesystem::path& path) : _path(path){};

private:
    std::filesystem::path _path;
};