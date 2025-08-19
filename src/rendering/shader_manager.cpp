#include "shader_manager.hpp"
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
          name_(name) {}

    void ManagedShader::bind() {
        if (shader_)
            shader_->bind();
    }

    void ManagedShader::unbind() {
        if (shader_)
            shader_->unbind();
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
        shader_->bind();
    }

    ShaderScope::~ShaderScope() {
        shader_->unbind();
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

        try {
            auto vert_path = getShaderPath(std::string(vert_file));
            auto frag_path = getShaderPath(std::string(frag_file));

            auto shader = std::make_shared<Shader>(
                vert_path.string().c_str(),
                frag_path.string().c_str(),
                create_buffer);

            return ManagedShader(shader, name);

        } catch (const std::exception& e) {
            return std::unexpected(ShaderError{
                std::format("Failed to load shader '{}': {}", name, e.what()), loc});
        }
    }

} // namespace gs::rendering
