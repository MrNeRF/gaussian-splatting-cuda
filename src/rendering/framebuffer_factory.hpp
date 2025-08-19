#pragma once
#include "framebuffer.hpp"
#include <memory>

namespace gs::rendering {
    enum class FrameBufferMode {
        CPU,
        CUDA_INTEROP
    };

    // Get the preferred mode based on compile-time configuration
    FrameBufferMode getPreferredFrameBufferMode();

    // Check if interop is available at compile time
    bool isInteropAvailable();

    // Create a framebuffer with the specified mode
    std::shared_ptr<FrameBuffer> createFrameBuffer(FrameBufferMode preferred = FrameBufferMode::CPU);
} // namespace gs::rendering