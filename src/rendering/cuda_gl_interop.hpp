#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <optional>
#include <torch/torch.h>

// Forward declare GLuint to avoid including OpenGL headers
typedef unsigned int GLuint;

// Include framebuffer after forward declarations
#include "framebuffer.hpp"

namespace gs::rendering {

    // Forward declaration for CUDA graphics resource
    struct CudaGraphicsResourceDeleter {
        void operator()(void* resource) const;
    };

    using CudaGraphicsResourcePtr = std::unique_ptr<void, CudaGraphicsResourceDeleter>;

    // Template declaration only - no implementation
    template <bool EnableInterop>
    class CudaGLInteropTextureImpl;

    // Full specialization for disabled interop
    template <>
    class CudaGLInteropTextureImpl<false> {
        GLuint texture_id_ = 0;
        int width_ = 0;
        int height_ = 0;

    public:
        CudaGLInteropTextureImpl() = default;
        ~CudaGLInteropTextureImpl();

        Result<void> init(int width, int height);
        Result<void> resize(int new_width, int new_height);
        Result<void> updateFromTensor(const torch::Tensor& image);
        GLuint getTextureID() const { return texture_id_; }

    private:
        void cleanup();
    };

    // Full specialization for enabled interop
    template <>
    class CudaGLInteropTextureImpl<true> {
        GLuint texture_id_ = 0;
        CudaGraphicsResourcePtr cuda_resource_;
        int width_ = 0;
        int height_ = 0;
        bool is_registered_ = false;

    public:
        CudaGLInteropTextureImpl();
        ~CudaGLInteropTextureImpl();

        Result<void> init(int width, int height);
        Result<void> resize(int new_width, int new_height);
        Result<void> updateFromTensor(const torch::Tensor& image);
        GLuint getTextureID() const { return texture_id_; }

    private:
        void cleanup();
    };

    // Type alias based on compile-time configuration
#ifdef CUDA_GL_INTEROP_ENABLED
    using CudaGLInteropTexture = CudaGLInteropTextureImpl<true>;
#else
    using CudaGLInteropTexture = CudaGLInteropTextureImpl<false>;
#endif

    // Modified FrameBuffer to support interop
    class InteropFrameBuffer : public FrameBuffer {
        std::optional<CudaGLInteropTexture> interop_texture_;
        bool use_interop_;

    public:
        explicit InteropFrameBuffer(bool use_interop = true);

        Result<void> uploadFromCUDA(const torch::Tensor& cuda_image);

        GLuint getInteropTexture() const {
            return use_interop_ && interop_texture_ ? interop_texture_->getTextureID() : getFrameTexture();
        }

        void resize(int new_width, int new_height) override;
    };

} // namespace gs::rendering