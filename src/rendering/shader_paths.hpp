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
#ifdef RENDERING_SOURCE_SHADER_PATH
        std::filesystem::path source_path = std::filesystem::path(RENDERING_SOURCE_SHADER_PATH) / shader_name;
        if (std::filesystem::exists(source_path)) {
            return source_path;
        }
#endif

        // Last resort - try relative to current file
        std::filesystem::path fallback_path = std::filesystem::path(__FILE__).parent_path() / "resources" / "shaders" / shader_name;
        if (std::filesystem::exists(fallback_path)) {
            return fallback_path;
        }

        throw std::runtime_error("Cannot find shader: " + shader_name);
    }

} // namespace gs::rendering