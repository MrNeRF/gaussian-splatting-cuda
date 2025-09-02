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
        LOG_TRACE("ManagedShader created: {} (shader ptr: {})", name_, static_cast<void*>(shader_.get()));
    }

    Result<void> ManagedShader::bind() {
        if (!shader_) {
            LOG_ERROR("Shader '{}' not initialized (nullptr)", name_);
            return std::unexpected(std::format("Shader '{}' not initialized", name_));
        }

        try {
            LOG_TRACE("Attempting to bind shader: {} (program ID: {})", name_, shader_->programID());

            // Check current GL state before binding
            GLint current_program = 0;
            glGetIntegerv(GL_CURRENT_PROGRAM, &current_program);
            if (current_program != 0) {
                LOG_TRACE("Switching from program {} to program {}", current_program, shader_->programID());
            }

            shader_->bind();

            // Verify binding was successful
            glGetIntegerv(GL_CURRENT_PROGRAM, &current_program);
            if (current_program != static_cast<GLint>(shader_->programID())) {
                LOG_ERROR("Failed to bind shader '{}': expected program {}, but current is {}",
                          name_, shader_->programID(), current_program);
                return std::unexpected(std::format("Failed to bind shader '{}': program mismatch", name_));
            }

            LOG_TRACE("Successfully bound shader: {} (program ID: {})", name_, shader_->programID());
            return {};
        } catch (const std::exception& e) {
            LOG_ERROR("Exception while binding shader '{}': {}", name_, e.what());
            return std::unexpected(std::format("Failed to bind shader '{}': {}", name_, e.what()));
        } catch (...) {
            LOG_ERROR("Unknown exception while binding shader '{}'", name_);
            return std::unexpected(std::format("Failed to bind shader '{}': unknown exception", name_));
        }
    }

    Result<void> ManagedShader::unbind() {
        if (!shader_) {
            LOG_ERROR("Shader '{}' not initialized (nullptr) during unbind", name_);
            return std::unexpected(std::format("Shader '{}' not initialized", name_));
        }

        try {
            LOG_TRACE("Attempting to unbind shader: {} (program ID: {})", name_, shader_->programID());

            // Check if this shader is currently bound
            GLint current_program = 0;
            glGetIntegerv(GL_CURRENT_PROGRAM, &current_program);
            if (current_program != static_cast<GLint>(shader_->programID())) {
                LOG_WARN("Unbinding shader '{}' (program {}) but current program is {}",
                         name_, shader_->programID(), current_program);
            }

            shader_->unbind();

            // Verify unbinding was successful
            glGetIntegerv(GL_CURRENT_PROGRAM, &current_program);
            if (current_program != 0) {
                LOG_WARN("After unbinding shader '{}', current program is {} instead of 0",
                         name_, current_program);
            }

            LOG_TRACE("Successfully unbound shader: {}", name_);
            return {};
        } catch (const std::exception& e) {
            LOG_ERROR("Exception while unbinding shader '{}': {}", name_, e.what());
            return std::unexpected(std::format("Failed to unbind shader '{}': {}", name_, e.what()));
        } catch (...) {
            LOG_ERROR("Unknown exception while unbinding shader '{}'", name_);
            return std::unexpected(std::format("Failed to unbind shader '{}': unknown exception", name_));
        }
    }

    Shader* ManagedShader::operator->() {
        if (!shader_) {
            LOG_ERROR("Attempting to dereference null shader pointer for '{}'", name_);
        }
        return shader_.get();
    }

    const Shader* ManagedShader::operator->() const {
        if (!shader_) {
            LOG_ERROR("Attempting to dereference null shader pointer for '{}' (const)", name_);
        }
        return shader_.get();
    }

    bool ManagedShader::valid() const {
        bool is_valid = (shader_ != nullptr);
        LOG_TRACE("Shader '{}' validity check: {}", name_, is_valid ? "valid" : "invalid");
        return is_valid;
    }

    ShaderScope::ShaderScope(ManagedShader& shader) : shader_(&shader) {
        LOG_TRACE("ShaderScope: Entering scope for shader '{}'",
                  shader_->name_.empty() ? "<unnamed>" : shader_->name_);

        if (!shader_->valid()) {
            LOG_ERROR("ShaderScope: Attempting to create scope for invalid shader '{}'",
                      shader_->name_.empty() ? "<unnamed>" : shader_->name_);
            bound_ = false;
            return;
        }

        if (auto result = shader_->bind(); result) {
            bound_ = true;
            LOG_TRACE("ShaderScope: Successfully bound shader in scope");
        } else {
            LOG_WARN("ShaderScope failed to bind shader: {}", result.error());
            bound_ = false;
        }
    }

    ShaderScope::~ShaderScope() {
        if (bound_) {
            LOG_TRACE("ShaderScope: Leaving scope, unbinding shader '{}'",
                      shader_->name_.empty() ? "<unnamed>" : shader_->name_);

            if (auto result = shader_->unbind(); !result) {
                LOG_ERROR("ShaderScope: Failed to unbind shader in destructor: {}", result.error());
            }
        } else {
            LOG_TRACE("ShaderScope: Leaving scope, shader was not bound");
        }
    }

    ManagedShader* ShaderScope::operator->() {
        if (!shader_) {
            LOG_ERROR("ShaderScope: Attempting to dereference null shader pointer");
        }
        return shader_;
    }

    ManagedShader& ShaderScope::operator*() {
        if (!shader_) {
            LOG_ERROR("ShaderScope: Attempting to dereference null shader pointer");
            throw std::runtime_error("ShaderScope: null shader pointer");
        }
        return *shader_;
    }

    ShaderResult<ManagedShader> load_shader(
        std::string_view name,
        std::string_view vert_file,
        std::string_view frag_file,
        bool create_buffer,
        std::source_location loc) {

        LOG_TIMER_TRACE("load_shader");
        LOG_DEBUG("Loading shader '{}' from {}:{}", name,
                  std::filesystem::path(loc.file_name()).filename().string(), loc.line());
        LOG_DEBUG("  Vertex shader: {}", vert_file);
        LOG_DEBUG("  Fragment shader: {}", frag_file);
        LOG_DEBUG("  Create buffer: {}", create_buffer ? "yes" : "no");

        try {
            // Get shader paths
            auto vert_path = getShaderPath(std::string(vert_file));
            auto frag_path = getShaderPath(std::string(frag_file));

            LOG_TRACE("Resolved vertex shader path: {}", vert_path.string());
            LOG_TRACE("Resolved fragment shader path: {}", frag_path.string());

            // Verify files exist
            if (!std::filesystem::exists(vert_path)) {
                LOG_ERROR("Vertex shader file not found: {}", vert_path.string());
                return std::unexpected(ShaderError{
                    std::format("Vertex shader file not found: {}", vert_path.string()), loc});
            }

            if (!std::filesystem::exists(frag_path)) {
                LOG_ERROR("Fragment shader file not found: {}", frag_path.string());
                return std::unexpected(ShaderError{
                    std::format("Fragment shader file not found: {}", frag_path.string()), loc});
            }

            // Check file sizes
            auto vert_size = std::filesystem::file_size(vert_path);
            auto frag_size = std::filesystem::file_size(frag_path);
            LOG_TRACE("Vertex shader file size: {} bytes", vert_size);
            LOG_TRACE("Fragment shader file size: {} bytes", frag_size);

            if (vert_size == 0) {
                LOG_ERROR("Vertex shader file is empty: {}", vert_path.string());
                return std::unexpected(ShaderError{
                    std::format("Vertex shader file is empty: {}", vert_path.string()), loc});
            }

            if (frag_size == 0) {
                LOG_ERROR("Fragment shader file is empty: {}", frag_path.string());
                return std::unexpected(ShaderError{
                    std::format("Fragment shader file is empty: {}", frag_path.string()), loc});
            }

            // Clear any existing OpenGL errors before shader creation
            GLenum gl_err;
            while ((gl_err = glGetError()) != GL_NO_ERROR) {
                LOG_WARN("Clearing pre-existing OpenGL error before shader creation: 0x{:x}", gl_err);
            }

            // Create shader
            auto shader = std::make_shared<Shader>(
                vert_path.string().c_str(),
                frag_path.string().c_str(),
                create_buffer);

            if (!shader) {
                LOG_ERROR("Failed to create shader object for '{}'", name);
                return std::unexpected(ShaderError{
                    std::format("Failed to create shader object for '{}'", name), loc});
            }

            // Verify shader program is valid
            GLuint program_id = shader->programID();
            if (program_id == 0) {
                LOG_ERROR("Created shader '{}' has invalid program ID (0)", name);
                return std::unexpected(ShaderError{
                    std::format("Created shader '{}' has invalid program ID", name), loc});
            }

            // Check for any OpenGL errors after creation
            gl_err = glGetError();
            if (gl_err != GL_NO_ERROR) {
                LOG_WARN("OpenGL error after creating shader '{}': 0x{:x}", name, gl_err);
            }

            LOG_INFO("Shader '{}' loaded successfully (program ID: {})", name, program_id);
            return ManagedShader(shader, name);

        } catch (const std::exception& e) {
            LOG_ERROR("Exception while loading shader '{}': {}", name, e.what());
            LOG_ERROR("  Called from: {}:{}",
                      std::filesystem::path(loc.file_name()).filename().string(), loc.line());
            return std::unexpected(ShaderError{
                std::format("Failed to load shader '{}': {}", name, e.what()), loc});
        } catch (...) {
            LOG_ERROR("Unknown exception while loading shader '{}'", name);
            LOG_ERROR("  Called from: {}:{}",
                      std::filesystem::path(loc.file_name()).filename().string(), loc.line());
            return std::unexpected(ShaderError{
                std::format("Failed to load shader '{}': unknown exception", name), loc});
        }
    }

} // namespace gs::rendering