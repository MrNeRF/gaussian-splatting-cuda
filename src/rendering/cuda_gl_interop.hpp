#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>

// Forward declare GLuint to avoid including OpenGL headers
typedef unsigned int GLuint;

// Include framebuffer after forward declarations
#include "framebuffer.hpp"

#ifdef CUDA_GL_INTEROP_ENABLED
// Only include cuda_gl_interop.h in implementation files, not headers
struct cudaGraphicsResource;
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;

namespace gs::rendering {

    class CudaGLInteropTexture {
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

    // Modified FrameBuffer to support interop
    class InteropFrameBuffer : public FrameBuffer {
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

} // namespace gs::rendering

#endif // CUDA_GL_INTEROP_ENABLED