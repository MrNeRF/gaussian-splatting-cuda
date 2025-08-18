#pragma once

#include <filesystem>
#include <string>

namespace gs::rendering {

    inline std::filesystem::path getShaderPath(const std::string& shader_name) {
        // Try build directory first
#ifdef RENDERING_SHADER_PATH
        std::filesystem::path build_path = std::filesystem::path(RENDERING_SHADER_PATH) / shader_name;
        if (std::filesystem::exists(build_path)) {
            return build_path;
        }
#endif

        // Fall back to source directory
#ifdef RENDERING_SOURCE_SHADER_PATH
        std::filesystem::path source_path = std::filesystem::path(RENDERING_SOURCE_SHADER_PATH) / shader_name;
        if (std::filesystem::exists(source_path)) {
            return source_path;
        }
#endif

        // Last resort - use PROJECT_ROOT_PATH
#ifdef PROJECT_ROOT_PATH
        return std::filesystem::path(PROJECT_ROOT_PATH) / "src/rendering/resources/shaders" / shader_name;
#else
        throw std::runtime_error("Cannot find shader: " + shader_name);
#endif
    }

} // namespace gs::rendering
