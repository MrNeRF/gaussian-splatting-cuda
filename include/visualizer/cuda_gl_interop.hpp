#pragma once

#include "config.h"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <torch/torch.h>

// Forward declare GLuint to avoid including OpenGL headers
typedef unsigned int GLuint;

// Include framebuffer after forward declarations
#include "visualizer/framebuffer.hpp"

#ifdef CUDA_GL_INTEROP_ENABLED
// Only include cuda_gl_interop.h in implementation files, not headers
struct cudaGraphicsResource;
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;
#endif

namespace gs {

#ifdef CUDA_GL_INTEROP_ENABLED

    class CudaGLInteropTexture {
    private:
        GLuint texture_id_;
        cudaGraphicsResource_t cuda_resource_;
        int width_;
        int height_;
        bool is_registered_;

    public:
        CudaGLInteropTexture();
        ~CudaGLInteropTexture();

        void init(int width, int height);
        void resize(int new_width, int new_height);
        void updateFromTensor(const torch::Tensor& image);
        GLuint getTextureID() const { return texture_id_; }

    private:
        void cleanup();
    };

    // Forward declaration - implemented in CUDA file
    torch::Tensor convertToRGBA8(const torch::Tensor& input, bool flip_vertical = false);

    // Modified FrameBuffer to support interop
    class InteropFrameBuffer : public FrameBuffer {
    private:
        CudaGLInteropTexture interop_texture_;
        bool use_interop_;

    public:
        explicit InteropFrameBuffer(bool use_interop = true);

        void uploadFromCUDA(const torch::Tensor& cuda_image);

        GLuint getInteropTexture() const {
            return use_interop_ ? interop_texture_.getTextureID() : getFrameTexture();
        }

        void resize(int new_width, int new_height) override;
    };

#else // CUDA_GL_INTEROP_ENABLED not defined

    // Stub implementations when interop is not available
    class CudaGLInteropTexture {
    public:
        CudaGLInteropTexture() = default;
        void init(int width, int height) {}
        void resize(int new_width, int new_height) {}
        void updateFromTensor(const torch::Tensor& image) {}
        GLuint getTextureID() const { return 0; }
    };

    class InteropFrameBuffer : public FrameBuffer {
    public:
        explicit InteropFrameBuffer(bool use_interop = false) : FrameBuffer() {}
        void uploadFromCUDA(const torch::Tensor& cuda_image) {
            // Fallback to CPU copy
            auto cpu_image = cuda_image.to(torch::kCPU).contiguous();
            if (cpu_image.dtype() != torch::kUInt8) {
                cpu_image = (cpu_image.clamp(0.0f, 1.0f) * 255.0f).to(torch::kUInt8);
            }
            uploadImage(cpu_image.data_ptr<unsigned char>(),
                        cuda_image.size(1), cuda_image.size(0));
        }
        GLuint getInteropTexture() const { return getFrameTexture(); }
    };

#endif // CUDA_GL_INTEROP_ENABLED

} // namespace gs