#pragma once

#include <expected>
#include <format>
#include <glad/glad.h>
#include <source_location>
#include <span>

namespace gs::rendering {

    struct GLError {
        std::string message;
        std::source_location location;

        std::string what() const {
            return std::format("[{}:{}] {}",
                               std::filesystem::path(location.file_name()).filename().string(),
                               location.line(), message);
        }
    };

    template <typename T>
    using GLResult = std::expected<T, GLError>;

    // RAII wrappers for OpenGL resources
    template <auto Deleter>
    class GLResource {
        GLuint id_ = 0;

    public:
        GLResource() = default;
        explicit GLResource(GLuint id) : id_(id) {}
        ~GLResource() {
            if (id_)
                Deleter(1, &id_);
        }

        // Move only
        GLResource(GLResource&& other) noexcept : id_(std::exchange(other.id_, 0)) {}
        GLResource& operator=(GLResource&& other) noexcept {
            if (this != &other) {
                if (id_)
                    Deleter(1, &id_);
                id_ = std::exchange(other.id_, 0);
            }
            return *this;
        }

        GLResource(const GLResource&) = delete;
        GLResource& operator=(const GLResource&) = delete;

        GLuint get() const { return id_; }
        GLuint* ptr() { return &id_; }
        operator GLuint() const { return id_; }
        explicit operator bool() const { return id_ != 0; }
    };

    // Specific resource types
    using VAO = GLResource<glDeleteVertexArrays>;
    using VBO = GLResource<glDeleteBuffers>;
    using EBO = GLResource<glDeleteBuffers>;
    using Texture = GLResource<glDeleteTextures>;
    using FBO = GLResource<glDeleteFramebuffers>;

    // Factory functions with error handling
    inline GLResult<VAO> create_vao(std::source_location loc = std::source_location::current()) {
        GLuint id;
        glGenVertexArrays(1, &id);
        if (glGetError() != GL_NO_ERROR || id == 0) {
            return std::unexpected(GLError{"Failed to create VAO", loc});
        }
        return VAO(id);
    }

    inline GLResult<VBO> create_vbo(std::source_location loc = std::source_location::current()) {
        GLuint id;
        glGenBuffers(1, &id);
        if (glGetError() != GL_NO_ERROR || id == 0) {
            return std::unexpected(GLError{"Failed to create VBO", loc});
        }
        return VBO(id);
    }

    // Scoped binders
    template <GLenum Target>
    class BufferBinder {
        GLint prev_;
        static constexpr GLenum query = (Target == GL_ARRAY_BUFFER) ? GL_ARRAY_BUFFER_BINDING : GL_ELEMENT_ARRAY_BUFFER_BINDING;

    public:
        explicit BufferBinder(const VBO& vbo) {
            glGetIntegerv(query, &prev_);
            glBindBuffer(Target, vbo);
        }
        ~BufferBinder() { glBindBuffer(Target, prev_); }
    };

    class VAOBinder {
        GLint prev_;

    public:
        explicit VAOBinder(const VAO& vao) {
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prev_);
            glBindVertexArray(vao);
        }
        ~VAOBinder() { glBindVertexArray(prev_); }
    };

    // Helper for vertex attribute setup
    struct VertexAttribute {
        GLuint index;
        GLint size;
        GLenum type;
        GLboolean normalized = GL_FALSE;
        GLsizei stride = 0;
        const void* offset = nullptr;
        GLuint divisor = 0; // For instancing

        void apply() const {
            glEnableVertexAttribArray(index);
            glVertexAttribPointer(index, size, type, normalized, stride, offset);
            if (divisor > 0)
                glVertexAttribDivisor(index, divisor);
        }
    };

    // Convenience function for buffer data upload
    template <typename T>
    void upload_buffer(GLenum target, std::span<const T> data, GLenum usage = GL_STATIC_DRAW) {
        glBufferData(target, data.size_bytes(), data.data(), usage);
    }

} // namespace gs::rendering
