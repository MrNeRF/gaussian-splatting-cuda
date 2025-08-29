/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "shader_manager.hpp"
#include "core/logger.hpp"
#include "shader_paths.hpp"
#include <filesystem>

namespace gs::rendering {

    std::string ShaderError::what() const {
        return std::format("[{}:{}] {}",
                           std::filesystem::path(location.file_name()).filename().string(),
                           location.line(), message);
    }

    ManagedShader::ManagedShader(std::shared_ptr<Shader> shader, std::string_view name)
        : shader_(shader),
          name_(name) {
        LOG_TRACE("ManagedShader created: {}", name_);
    }

    Result<void> ManagedShader::bind() {
        if (!shader_) {
            LOG_ERROR("Shader '{}' not initialized", name_);
            return std::unexpected("Shader not initialized");
        }

        try {
            shader_->bind();
            LOG_TRACE("Bound shader: {}", name_);
            return {};
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to bind shader '{}': {}", name_, e.what());
            return std::unexpected(std::format("Failed to bind shader '{}': {}", name_, e.what()));
        }
    }

    Result<void> ManagedShader::unbind() {
        if (!shader_) {
            LOG_ERROR("Shader '{}' not initialized", name_);
            return std::unexpected("Shader not initialized");
        }

        try {
            shader_->unbind();
            LOG_TRACE("Unbound shader: {}", name_);
            return {};
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to unbind shader '{}': {}", name_, e.what());
            return std::unexpected(std::format("Failed to unbind shader '{}': {}", name_, e.what()));
        }
    }

    Shader* ManagedShader::operator->() {
        return shader_.get();
    }

    const Shader* ManagedShader::operator->() const {
        return shader_.get();
    }

    bool ManagedShader::valid() const {
        return shader_ != nullptr;
    }

    ShaderScope::ShaderScope(ManagedShader& shader) : shader_(&shader) {
        if (auto result = shader_->bind(); result) {
            bound_ = true;
        } else {
            LOG_WARN("ShaderScope failed to bind shader");
        }
    }

    ShaderScope::~ShaderScope() {
        if (bound_) {
            shader_->unbind();
        }
    }

    ManagedShader* ShaderScope::operator->() {
        return shader_;
    }

    ManagedShader& ShaderScope::operator*() {
        return *shader_;
    }

    ShaderResult<ManagedShader> load_shader(
        std::string_view name,
        std::string_view vert_file,
        std::string_view frag_file,
        bool create_buffer,
        std::source_location loc) {

        LOG_TIMER_TRACE("load_shader");
        LOG_DEBUG("Loading shader '{}': vertex={}, fragment={}", name, vert_file, frag_file);

        try {
            auto vert_path = getShaderPath(std::string(vert_file));
            auto frag_path = getShaderPath(std::string(frag_file));

            LOG_TRACE("Vertex shader path: {}", vert_path.string());
            LOG_TRACE("Fragment shader path: {}", frag_path.string());

            auto shader = std::make_shared<Shader>(
                vert_path.string().c_str(),
                frag_path.string().c_str(),
                create_buffer);

            LOG_INFO("Shader '{}' loaded successfully", name);
            return ManagedShader(shader, name);

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load shader '{}': {}", name, e.what());
            return std::unexpected(ShaderError{
                std::format("Failed to load shader '{}': {}", name, e.what()), loc});
        }
    }

} // namespace gs::rendering