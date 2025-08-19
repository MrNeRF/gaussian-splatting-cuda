#include "framebuffer_factory.hpp"
#include <format>
#include <iostream>

#ifdef CUDA_GL_INTEROP_ENABLED
#include "cuda_gl_interop.hpp"
#endif

namespace gs::rendering {

    std::shared_ptr<FrameBuffer> createFrameBuffer(FrameBufferMode preferred) {
        if (preferred == FrameBufferMode::CUDA_INTEROP && isInteropAvailable()) {
#ifdef CUDA_GL_INTEROP_ENABLED
            try {
                auto interop_fb = std::make_shared<InteropFrameBuffer>(true);
                std::cout << "CUDA-OpenGL interop framebuffer initialized successfully" << std::endl;
                return interop_fb;
            } catch (const std::exception& e) {
                std::cerr << "Failed to initialize interop framebuffer: " << e.what() << std::endl;
                std::cerr << "Falling back to standard framebuffer" << std::endl;
            }
#endif
        }
        return std::make_shared<FrameBuffer>();
    }
} // namespace gs::rendering
