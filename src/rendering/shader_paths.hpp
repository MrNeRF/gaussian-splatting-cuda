#pragma once

#include "core/logger.hpp"
#include <filesystem>
#include <string>

namespace gs::rendering {

    inline std::filesystem::path getShaderPath(const std::string& shader_name) {
        LOG_TIMER_TRACE("getShaderPath");
        LOG_TRACE("Looking for shader: {}", shader_name);

        // Try build directory first
#ifdef SHADER_PATH
        std::filesystem::path build_path = std::filesystem::path(SHADER_PATH) / shader_name;
        if (std::filesystem::exists(build_path)) {
            LOG_DEBUG("Found shader in build directory: {}", build_path.string());
            return build_path;
        }
        LOG_TRACE("Shader not found in build directory: {}", build_path.string());
#endif

        // Fall back to source directory
#ifdef RENDERING_SOURCE_SHADER_PATH
        std::filesystem::path source_path = std::filesystem::path(RENDERING_SOURCE_SHADER_PATH) / shader_name;
        if (std::filesystem::exists(source_path)) {
            LOG_DEBUG("Found shader in source directory: {}", source_path.string());
            return source_path;
        }
        LOG_TRACE("Shader not found in source directory: {}", source_path.string());
#endif

        // Last resort - try relative to current file
        std::filesystem::path fallback_path = std::filesystem::path(__FILE__).parent_path() / "resources" / "shaders" / shader_name;
        if (std::filesystem::exists(fallback_path)) {
            LOG_DEBUG("Found shader in fallback path: {}", fallback_path.string());
            return fallback_path;
        }

        LOG_ERROR("Cannot find shader: {}", shader_name);
        throw std::runtime_error("Cannot find shader: " + shader_name);
    }

} // namespace gs::rendering