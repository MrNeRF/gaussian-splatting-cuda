#pragma once

#include "core/logger.hpp"
#include <expected>
#include <format>
#include <string>

namespace gs::rendering {

    // Error types for rendering operations
    enum class RenderError {
        NotInitialized,
        InvalidViewport,
        ShaderCompilationFailed,
        ResourceCreationFailed,
        CudaInteropFailed,
        InvalidInput,
        OpenGLError
    };

    // Convert error enum to string
    inline std::string to_string(RenderError error) {
        switch (error) {
        case RenderError::NotInitialized: return "Rendering engine not initialized";
        case RenderError::InvalidViewport: return "Invalid viewport dimensions";
        case RenderError::ShaderCompilationFailed: return "Shader compilation failed";
        case RenderError::ResourceCreationFailed: return "Failed to create OpenGL resources";
        case RenderError::CudaInteropFailed: return "CUDA-OpenGL interop failed";
        case RenderError::InvalidInput: return "Invalid input data";
        case RenderError::OpenGLError: return "OpenGL error occurred";
        default: return "Unknown error";
        }
    }

    // Extended error information
    struct RenderErrorInfo {
        RenderError code;
        std::string details;

        RenderErrorInfo(RenderError c, std::string_view d = "")
            : code(c),
              details(d) {
            if (!details.empty()) {
                LOG_ERROR("RenderError::{} - {}", to_string(code), details);
            } else {
                LOG_ERROR("RenderError::{}", to_string(code));
            }
        }

        std::string message() const {
            if (details.empty()) {
                return to_string(code);
            }
            return std::format("{}: {}", to_string(code), details);
        }
    };

    // Result type for rendering operations
    template <typename T>
    using Result = std::expected<T, RenderErrorInfo>;

    // Helper to create error results
    template <typename T>
    inline Result<T> make_error(RenderError code, std::string_view details = "") {
        return std::unexpected(RenderErrorInfo{code, std::string(details)});
    }

} // namespace gs::rendering