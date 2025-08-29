/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "shader.hpp"
#include <expected>
#include <format>
#include <memory>
#include <source_location>
#include <string>

namespace gs::rendering {

    template <typename T>
    using Result = std::expected<T, std::string>;

    struct ShaderError {
        std::string message;
        std::source_location location;

        std::string what() const;
    };

    template <typename T>
    using ShaderResult = std::expected<T, ShaderError>;

    class ManagedShader {
        std::shared_ptr<Shader> shader_;
        std::string name_;

    public:
        ManagedShader() = default;
        ManagedShader(std::shared_ptr<Shader> shader, std::string_view name);

        Result<void> bind();
        Result<void> unbind();

        // Delegate to existing shader with error location tracking
        template <typename T>
        Result<void> set(std::string_view uniform, const T& value,
                         std::source_location loc = std::source_location::current()) {
            try {
                shader_->set_uniform(std::string(uniform), value);
                return {};
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Shader '{}': {}", name_, e.what()));
            }
        }

        Shader* operator->();
        const Shader* operator->() const;
        bool valid() const;
    };

    // RAII scope guard - more natural syntax
    class ShaderScope {
        ManagedShader* shader_;
        bool bound_ = false;

    public:
        explicit ShaderScope(ManagedShader& shader);
        ~ShaderScope();
        ManagedShader* operator->();
        ManagedShader& operator*();
    };

    // Ultra-minimal shader loader
    ShaderResult<ManagedShader> load_shader(
        std::string_view name,
        std::string_view vert_file,
        std::string_view frag_file,
        bool create_buffer = false,
        std::source_location loc = std::source_location::current());

} // namespace gs::rendering