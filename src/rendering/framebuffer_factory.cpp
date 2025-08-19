#include "framebuffer_factory.hpp"
#include <iostream>

#ifdef CUDA_GL_INTEROP_ENABLED
#include "cuda_gl_interop.hpp"
#include <cuda_gl_interop.h>
#endif

namespace gs::rendering {

    FrameBufferMode getPreferredFrameBufferMode() {
#ifdef CUDA_GL_INTEROP_ENABLED
        return FrameBufferMode::CUDA_INTEROP;
#else
        return FrameBufferMode::CPU;
#endif
    }

    bool isInteropAvailable() {
#ifdef CUDA_GL_INTEROP_ENABLED
        return true;
#else
        return false;
#endif
    }

    std::shared_ptr<FrameBuffer> createFrameBuffer(FrameBufferMode preferred) {
#ifdef CUDA_GL_INTEROP_ENABLED
        if (preferred == FrameBufferMode::CUDA_INTEROP) {
            try {
                auto interop_fb = std::make_shared<InteropFrameBuffer>(true);
                std::cout << "CUDA-OpenGL interop framebuffer initialized successfully" << std::endl;
                return interop_fb;
            } catch (const std::exception& e) {
                std::cerr << "Failed to initialize interop framebuffer: " << e.what() << std::endl;
                std::cerr << "Falling back to standard framebuffer" << std::endl;
            }
        }
#endif
        return std::make_shared<FrameBuffer>();
    }
} // namespace gs::rendering