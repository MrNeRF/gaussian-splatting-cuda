#include "config.h"

// clang-format off
// CRITICAL: GLAD must be included before GLFW to avoid OpenGL header conflicts
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include "core/logger.hpp"
#include "cuda_gl_interop.hpp"
#include <format>

#ifdef CUDA_GL_INTEROP_ENABLED
// Only include CUDA GL interop when enabled
#include <cuda_gl_interop.h>
#endif

namespace gs::rendering {

    // Implementation for CudaGraphicsResourceDeleter
    void CudaGraphicsResourceDeleter::operator()(void* resource) const {
#ifdef CUDA_GL_INTEROP_ENABLED
        if (resource) {
            cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource_t>(resource));
            LOG_TRACE("Unregistered CUDA graphics resource");
        }
#endif
    }

    // Implementation for disabled interop version
    CudaGLInteropTextureImpl<false>::~CudaGLInteropTextureImpl() {
        cleanup();
    }

    Result<void> CudaGLInteropTextureImpl<false>::init(int width, int height) {
        LOG_TIMER_TRACE("CudaGLInteropTextureImpl<false>::init");
        LOG_DEBUG("Initializing non-interop texture: {}x{}", width, height);

        width_ = width;
        height_ = height;

        // Create regular OpenGL texture
        glGenTextures(1, &texture_id_);
        glBindTexture(GL_TEXTURE_2D, texture_id_);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glBindTexture(GL_TEXTURE_2D, 0);

        GLenum gl_err = glGetError();
        if (gl_err != GL_NO_ERROR) {
            cleanup();
            LOG_ERROR("OpenGL error during texture creation: {}", gl_err);
            return std::unexpected(std::format("OpenGL error during texture creation: {}", gl_err));
        }

        LOG_DEBUG("Non-interop texture created successfully");
        return {};
    }

    Result<void> CudaGLInteropTextureImpl<false>::resize(int new_width, int new_height) {
        if (width_ != new_width || height_ != new_height) {
            LOG_TRACE("Resizing non-interop texture from {}x{} to {}x{}",
                      width_, height_, new_width, new_height);
            return init(new_width, new_height);
        }
        return {};
    }

    Result<void> CudaGLInteropTextureImpl<false>::updateFromTensor(const torch::Tensor& image) {
        // CPU fallback - this should not be called for non-interop version
        LOG_ERROR("CUDA-GL interop not available - use regular framebuffer upload");
        return std::unexpected("CUDA-GL interop not available - use regular framebuffer upload");
    }

    void CudaGLInteropTextureImpl<false>::cleanup() {
        if (texture_id_ != 0) {
            LOG_TRACE("Cleaning up non-interop texture");
            glDeleteTextures(1, &texture_id_);
            texture_id_ = 0;
        }
    }

#ifdef CUDA_GL_INTEROP_ENABLED
    // Full implementation for when interop is enabled
    CudaGLInteropTextureImpl<true>::CudaGLInteropTextureImpl()
        : texture_id_(0),
          cuda_resource_(nullptr),
          width_(0),
          height_(0),
          is_registered_(false) {
        LOG_DEBUG("Creating CUDA-GL interop texture");
    }

    CudaGLInteropTextureImpl<true>::~CudaGLInteropTextureImpl() {
        cleanup();
    }

    Result<void> CudaGLInteropTextureImpl<true>::init(int width, int height) {
        LOG_TIMER("CudaGLInteropTextureImpl<true>::init");
        LOG_INFO("Initializing CUDA-GL interop texture: {}x{}", width, height);

        // Clean up any existing resources
        cleanup();

        width_ = width;
        height_ = height;

        // Create OpenGL texture
        glGenTextures(1, &texture_id_);
        glBindTexture(GL_TEXTURE_2D, texture_id_);

        // Set texture parameters BEFORE allocating storage
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Allocate texture storage (RGBA for better alignment)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        // CRITICAL: Unbind texture before registering with CUDA
        glBindTexture(GL_TEXTURE_2D, 0);

        // Check OpenGL errors
        GLenum gl_err = glGetError();
        if (gl_err != GL_NO_ERROR) {
            cleanup();
            LOG_ERROR("OpenGL error during texture creation: {}", gl_err);
            return std::unexpected(std::format("OpenGL error during texture creation: {}", gl_err));
        }

        // Clear any previous CUDA errors
        cudaGetLastError();

        // Register texture with CUDA
        cudaGraphicsResource_t raw_resource;
        cudaError_t err = cudaGraphicsGLRegisterImage(
            &raw_resource, texture_id_, GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsWriteDiscard);

        if (err != cudaSuccess) {
            cleanup();
            LOG_ERROR("Failed to register OpenGL texture with CUDA: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to register OpenGL texture with CUDA: {}",
                                               cudaGetErrorString(err)));
        }

        cuda_resource_.reset(raw_resource);
        is_registered_ = true;

        LOG_INFO("CUDA-GL interop texture initialized successfully");
        return {};
    }

    Result<void> CudaGLInteropTextureImpl<true>::resize(int new_width, int new_height) {
        if (width_ != new_width || height_ != new_height) {
            LOG_DEBUG("Resizing CUDA-GL interop texture from {}x{} to {}x{}",
                      width_, height_, new_width, new_height);
            return init(new_width, new_height);
        }
        return {};
    }

    Result<void> CudaGLInteropTextureImpl<true>::updateFromTensor(const torch::Tensor& image) {
        LOG_TIMER_TRACE("CudaGLInteropTextureImpl<true>::updateFromTensor");

        if (!is_registered_) {
            LOG_ERROR("Texture not initialized");
            return std::unexpected("Texture not initialized");
        }

        // Ensure tensor is CUDA, float32, and [H, W, C] format
        if (!image.is_cuda()) {
            LOG_ERROR("Image must be on CUDA");
            return std::unexpected("Image must be on CUDA");
        }
        if (image.dim() != 3) {
            LOG_ERROR("Image must be [H, W, C], got {} dimensions", image.dim());
            return std::unexpected("Image must be [H, W, C]");
        }
        if (image.size(2) != 3 && image.size(2) != 4) {
            LOG_ERROR("Image must have 3 or 4 channels, got {}", image.size(2));
            return std::unexpected("Image must have 3 or 4 channels");
        }

        const int h = image.size(0);
        const int w = image.size(1);
        const int c = image.size(2);

        LOG_TRACE("Updating from tensor: {}x{}x{}", h, w, c);

        // Resize if needed
        if (auto result = resize(w, h); !result) {
            return result;
        }

        // Map CUDA resource
        auto raw_resource = static_cast<cudaGraphicsResource_t>(cuda_resource_.get());
        cudaError_t err = cudaGraphicsMapResources(1, &raw_resource, 0);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to map CUDA resource: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to map CUDA resource: {}",
                                               cudaGetErrorString(err)));
        }

        // RAII unmap guard
        struct UnmapGuard {
            cudaGraphicsResource_t* resource;
            ~UnmapGuard() {
                if (resource) {
                    cudaGraphicsUnmapResources(1, resource, 0);
                    LOG_TRACE("Unmapped CUDA resource");
                }
            }
        } unmap_guard{&raw_resource};

        // Get CUDA array from mapped resource
        cudaArray_t cuda_array;
        err = cudaGraphicsSubResourceGetMappedArray(&cuda_array, raw_resource, 0, 0);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to get CUDA array: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to get CUDA array: {}",
                                               cudaGetErrorString(err)));
        }

        // Convert to RGBA uint8 if needed
        torch::Tensor rgba_image;
        if (c == 3) {
            // Add alpha channel
            rgba_image = torch::cat({image,
                                     torch::ones({h, w, 1}, image.options())},
                                    2);
            LOG_TRACE("Added alpha channel to image");
        } else {
            rgba_image = image;
        }

        // Ensure proper format (uint8)
        if (rgba_image.dtype() != torch::kUInt8) {
            rgba_image = (rgba_image.clamp(0.0f, 1.0f) * 255.0f).to(torch::kUInt8);
            LOG_TRACE("Converted image to uint8");
        }

        // Make contiguous
        rgba_image = rgba_image.contiguous();

        // Copy to CUDA array
        err = cudaMemcpy2DToArray(
            cuda_array,
            0, 0, // offset
            rgba_image.data_ptr<uint8_t>(),
            w * 4, // pitch (RGBA = 4 bytes per pixel)
            w * 4, // width in bytes
            h,     // height
            cudaMemcpyDeviceToDevice);

        if (err != cudaSuccess) {
            LOG_ERROR("Failed to copy to CUDA array: {}", cudaGetErrorString(err));
            return std::unexpected(std::format("Failed to copy to CUDA array: {}",
                                               cudaGetErrorString(err)));
        }

        // Synchronize to ensure copy is complete
        cudaDeviceSynchronize();

        LOG_TRACE("Successfully updated texture from CUDA tensor");
        return {};
    }

    void CudaGLInteropTextureImpl<true>::cleanup() {
        LOG_TRACE("Cleaning up CUDA-GL interop texture");
        cuda_resource_.reset();
        is_registered_ = false;

        if (texture_id_ != 0) {
            glDeleteTextures(1, &texture_id_);
            texture_id_ = 0;
        }
    }
#endif // CUDA_GL_INTEROP_ENABLED

    // InteropFrameBuffer implementation
    InteropFrameBuffer::InteropFrameBuffer(bool use_interop)
        : FrameBuffer(),
          use_interop_(use_interop) {
        LOG_DEBUG("Creating InteropFrameBuffer with interop: {}", use_interop);

        if (use_interop_) {
            interop_texture_.emplace();
            if (auto result = interop_texture_->init(width, height); !result) {
                LOG_WARN("Failed to initialize CUDA-GL interop: {}", result.error());
                LOG_INFO("Falling back to CPU copy mode");
                interop_texture_.reset();
                use_interop_ = false;
            }
        }
    }

    Result<void> InteropFrameBuffer::uploadFromCUDA(const torch::Tensor& cuda_image) {
        LOG_TIMER_TRACE("InteropFrameBuffer::uploadFromCUDA");

        if (!use_interop_ || !interop_texture_) {
            // Fallback to CPU copy
            LOG_TRACE("Using CPU fallback for CUDA upload");
            auto cpu_image = cuda_image;
            if (cuda_image.is_cuda()) {
                cpu_image = cuda_image.to(torch::kCPU);
            }
            cpu_image = cpu_image.contiguous();

            // Handle both [H, W, C] and [C, H, W] formats
            torch::Tensor formatted;
            if (cpu_image.size(-1) == 3 || cpu_image.size(-1) == 4) {
                // Already [H, W, C]
                formatted = cpu_image;
            } else {
                // Convert [C, H, W] to [H, W, C]
                formatted = cpu_image.permute({1, 2, 0}).contiguous();
            }

            // Convert to uint8 if needed
            if (formatted.dtype() != torch::kUInt8) {
                formatted = (formatted.clamp(0.0f, 1.0f) * 255.0f).to(torch::kUInt8);
            }

            uploadImage(formatted.data_ptr<unsigned char>(),
                        formatted.size(1), formatted.size(0));
            return {};
        }

        // Direct CUDA update
        LOG_TRACE("Using direct CUDA-GL interop update");
        auto result = interop_texture_->updateFromTensor(cuda_image);
        if (!result) {
            LOG_WARN("CUDA-GL interop update failed: {}", result.error());
            LOG_INFO("Falling back to CPU copy");
            use_interop_ = false;
            interop_texture_.reset();
            return uploadFromCUDA(cuda_image); // Retry with CPU fallback
        }
        return {};
    }

    void InteropFrameBuffer::resize(int new_width, int new_height) {
        LOG_TRACE("Resizing InteropFrameBuffer from {}x{} to {}x{}", width, height, new_width, new_height);
        FrameBuffer::resize(new_width, new_height);
        if (use_interop_ && interop_texture_) {
            if (auto result = interop_texture_->resize(new_width, new_height); !result) {
                LOG_WARN("Failed to resize interop texture: {}", result.error());
                use_interop_ = false;
                interop_texture_.reset();
            }
        }
    }

} // namespace gs::rendering