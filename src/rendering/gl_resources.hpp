#pragma once

#include <expected>
#include <filesystem>
#include <glad/glad.h>
#include <source_location>
#include <span>
#include <utility>

namespace gs::rendering {

    // Consistent error handling
    template <typename T>
    using Result = std::expected<T, std::string>;

    // RAII wrappers for OpenGL resources
    template <typename Deleter>
    class GLResource {
        GLuint id_ = 0;

    public:
        GLResource() = default;
        explicit GLResource(GLuint id) : id_(id) {}
        ~GLResource() {
            if (id_) {
                Deleter deleter;
                deleter(1, &id_);
            }
        }

        // Move only
        GLResource(GLResource&& other) noexcept : id_(std::exchange(other.id_, 0)) {}
        GLResource& operator=(GLResource&& other) noexcept {
            if (this != &other) {
                if (id_) {
                    Deleter deleter;
                    deleter(1, &id_);
                }
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

    // Deleter functors
    struct VAODeleter {
        void operator()(GLsizei n, const GLuint* arrays) const {
            glDeleteVertexArrays(n, arrays);
        }
    };

    struct BufferDeleter {
        void operator()(GLsizei n, const GLuint* buffers) const {
            glDeleteBuffers(n, buffers);
        }
    };

    struct TextureDeleter {
        void operator()(GLsizei n, const GLuint* textures) const {
            glDeleteTextures(n, textures);
        }
    };

    struct FramebufferDeleter {
        void operator()(GLsizei n, const GLuint* framebuffers) const {
            glDeleteFramebuffers(n, framebuffers);
        }
    };

    // Specific resource types
    using VAO = GLResource<VAODeleter>;
    using VBO = GLResource<BufferDeleter>;
    using EBO = GLResource<BufferDeleter>;
    using Texture = GLResource<TextureDeleter>;
    using FBO = GLResource<FramebufferDeleter>;

    // Factory functions with error handling
    inline Result<VAO> create_vao() {
        GLuint id;
        glGenVertexArrays(1, &id);
        if (glGetError() != GL_NO_ERROR || id == 0) {
            return std::unexpected("Failed to create VAO");
        }
        return VAO(id);
    }

    inline Result<VBO> create_vbo() {
        GLuint id;
        glGenBuffers(1, &id);
        if (glGetError() != GL_NO_ERROR || id == 0) {
            return std::unexpected("Failed to create VBO");
        }
        return VBO(id);
    }

    // Scoped binders
    template <GLenum Target>
    class BufferBinder {
        GLint prev_;
        static constexpr GLenum query = []() constexpr->GLenum {
            if (Target == GL_ARRAY_BUFFER)
                return GL_ARRAY_BUFFER_BINDING;
            if (Target == GL_ELEMENT_ARRAY_BUFFER)
                return GL_ELEMENT_ARRAY_BUFFER_BINDING;
            if (Target == GL_UNIFORM_BUFFER)
                return GL_UNIFORM_BUFFER_BINDING;
            if (Target == GL_TEXTURE_BUFFER)
                return GL_TEXTURE_BUFFER_BINDING;
            if (Target == GL_COPY_READ_BUFFER)
                return GL_COPY_READ_BUFFER_BINDING;
            if (Target == GL_COPY_WRITE_BUFFER)
                return GL_COPY_WRITE_BUFFER_BINDING;
            if (Target == GL_PIXEL_PACK_BUFFER)
                return GL_PIXEL_PACK_BUFFER_BINDING;
            if (Target == GL_PIXEL_UNPACK_BUFFER)
                return GL_PIXEL_UNPACK_BUFFER_BINDING;
            if (Target == GL_TRANSFORM_FEEDBACK_BUFFER)
                return GL_TRANSFORM_FEEDBACK_BUFFER_BINDING;
            return 0;
        }
        ();

    public:
        explicit BufferBinder(GLuint vbo) {
            glGetIntegerv(query, &prev_);
            glBindBuffer(Target, vbo);
        }
        ~BufferBinder() { glBindBuffer(Target, prev_); }
    };

    class VAOBinder {
        GLint prev_;

    public:
        explicit VAOBinder(GLuint vao) {
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
    void upload_buffer(GLenum target, const T* data, std::size_t count, GLenum usage = GL_STATIC_DRAW) {
        glBufferData(target, count * sizeof(T), data, usage);
    }

    // Overload for dynamic spans
    template <typename T>
    void upload_buffer(GLenum target, std::span<T> data, GLenum usage = GL_STATIC_DRAW) {
        glBufferData(target, data.size_bytes(), data.data(), usage);
    }

} // namespace gs::rendering