#include "framebuffer_factory.hpp"
#include "core/logger.hpp"
#include <format>

#ifdef CUDA_GL_INTEROP_ENABLED
#include "cuda_gl_interop.hpp"
#endif

namespace gs::rendering {

    std::shared_ptr<FrameBuffer> createFrameBuffer(FrameBufferMode preferred) {
        if (preferred == FrameBufferMode::CUDA_INTEROP && isInteropAvailable()) {
#ifdef CUDA_GL_INTEROP_ENABLED
            try {
                LOG_TIMER_TRACE("createFrameBuffer::CUDA_INTEROP");

                // Create with default size, will resize on first use if needed
                auto interop_fb = std::make_shared<InteropFrameBuffer>(true);
                LOG_INFO("CUDA-OpenGL interop framebuffer created (will size on first use)");
                return interop_fb;
            } catch (const std::exception& e) {
                LOG_WARN("Failed to initialize interop framebuffer: {}", e.what());
                LOG_INFO("Falling back to standard framebuffer");
            }
#endif
        }

        LOG_DEBUG("Creating standard CPU framebuffer");
        return std::make_shared<FrameBuffer>();
    }
} // namespace gs::rendering