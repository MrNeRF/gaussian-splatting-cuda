#pragma once

#include <filesystem>
#include <string>

namespace gs::visualizer {

    inline std::filesystem::path getShaderPath(const std::string& shader_name) {
// Try build directory first
#ifdef VISUALIZER_SHADER_PATH
        std::filesystem::path build_path = std::filesystem::path(VISUALIZER_SHADER_PATH) / shader_name;
        if (std::filesystem::exists(build_path)) {
            return build_path;
        }
#endif

// Fall back to source directory
#ifdef VISUALIZER_SOURCE_SHADER_PATH
        std::filesystem::path source_path = std::filesystem::path(VISUALIZER_SOURCE_SHADER_PATH) / shader_name;
        if (std::filesystem::exists(source_path)) {
            return source_path;
        }
#endif

// Last resort - use PROJECT_ROOT_PATH
#ifdef PROJECT_ROOT_PATH
        return std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/resources/shaders" / shader_name;
#else
        throw std::runtime_error("Cannot find shader: " + shader_name);
#endif
    }

    inline std::filesystem::path getAssetPath(const std::string& asset_name) {
// Try build directory first
#ifdef VISUALIZER_ASSET_PATH
        std::filesystem::path build_path = std::filesystem::path(VISUALIZER_ASSET_PATH) / asset_name;
        if (std::filesystem::exists(build_path)) {
            return build_path;
        }
#endif

// Fall back to source directory
#ifdef VISUALIZER_SOURCE_ASSET_PATH
        std::filesystem::path source_path = std::filesystem::path(VISUALIZER_SOURCE_ASSET_PATH) / asset_name;
        if (std::filesystem::exists(source_path)) {
            return source_path;
        }
#endif

// Last resort
#ifdef PROJECT_ROOT_PATH
        return std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/resources/assets" / asset_name;
#else
        throw std::runtime_error("Cannot find asset: " + asset_name);
#endif
    }

} // namespace gs::visualizer
