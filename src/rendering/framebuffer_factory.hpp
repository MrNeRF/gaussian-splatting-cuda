#pragma once
#include "framebuffer.hpp"
#include <memory>

namespace gs::rendering {
    enum class FrameBufferMode {
        CPU,
        CUDA_INTEROP
    };

    // Template-based feature detection instead of preprocessor
    template <bool EnableInterop = false>
    struct InteropCapability {
        static constexpr bool available = EnableInterop;
    };

    // Get the preferred mode based on compile-time configuration
    inline FrameBufferMode getPreferredFrameBufferMode() {
        // This will be set by CMake through a constexpr or config header
        constexpr bool cuda_interop_available =
#ifdef CUDA_GL_INTEROP_ENABLED
            true;
#else
            false;
#endif

        if constexpr (cuda_interop_available) {
            return FrameBufferMode::CUDA_INTEROP;
        }
        return FrameBufferMode::CPU;
    }

    // Check if interop is available at compile time
    inline bool isInteropAvailable() {
        return getPreferredFrameBufferMode() == FrameBufferMode::CUDA_INTEROP;
    }

    // Create a framebuffer with the specified mode
    std::shared_ptr<FrameBuffer> createFrameBuffer(FrameBufferMode preferred = FrameBufferMode::CPU);
} // namespace gs::rendering