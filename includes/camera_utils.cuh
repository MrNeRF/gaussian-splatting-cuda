#pragma once

#include "camera.cuh"
#include <nlohmann/json.hpp>

nlohmann::json camera_to_JSON(Camera cam);
