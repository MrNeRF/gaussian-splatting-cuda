#pragma once

#include <filesystem>
#include <string>

namespace gs::rendering {

    inline std::filesystem::path getShaderPath(const std::string& shader_name) {
        // Try build directory first
#ifdef SHADER_PATH
        std::filesystem::path build_path = std::filesystem::path(SHADER_PATH) / shader_name;
        if (std::filesystem::exists(build_path)) {
            return build_path;
        }
#endif

        // Fall back to source directory
        std::filesystem::path source_path = std::filesystem::path(PROJECT_ROOT_PATH) / "src/rendering/resources/shaders" / shader_name;
        if (std::filesystem::exists(source_path)) {
            return source_path;
        }

        // Last resort - check if we have RENDERING_SOURCE_SHADER_PATH defined
#ifdef RENDERING_SOURCE_SHADER_PATH
        std::filesystem::path rendering_source_path = std::filesystem::path(RENDERING_SOURCE_SHADER_PATH) / shader_name;
        if (std::filesystem::exists(rendering_source_path)) {
            return rendering_source_path;
        }
#endif

        throw std::runtime_error("Cannot find shader: " + shader_name);
    }

} // namespace gs::rendering