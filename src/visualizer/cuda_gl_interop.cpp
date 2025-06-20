#include "config.h"

#ifdef CUDA_GL_INTEROP_ENABLED

// clang-format off
// CRITICAL: GLAD must be included before GLFW to avoid OpenGL header conflicts
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

// Now include CUDA GL interop (after GLAD)
#include <cuda_gl_interop.h>

#include "visualizer/cuda_gl_interop.hpp"
#include <iostream>
#include <stdexcept>

namespace gs {

    CudaGLInteropTexture::CudaGLInteropTexture()
        : texture_id_(0),
          cuda_resource_(nullptr),
          width_(0),
          height_(0),
          is_registered_(false) {
    }

    CudaGLInteropTexture::~CudaGLInteropTexture() {
        cleanup();
    }

    void CudaGLInteropTexture::init(int width, int height) {
        // Clean up any existing resources
        cleanup();

        width_ = width;
        height_ = height;

        // Create OpenGL texture
        glGenTextures(1, &texture_id_);
        glBindTexture(GL_TEXTURE_2D, texture_id_);

        // Allocate texture storage (RGBA for better alignment)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Check OpenGL errors
        GLenum gl_err = glGetError();
        if (gl_err != GL_NO_ERROR) {
            cleanup();
            throw std::runtime_error("OpenGL error during texture creation: " +
                                     std::to_string(gl_err));
        }

        // Register texture with CUDA
        cudaError_t err = cudaGraphicsGLRegisterImage(
            &cuda_resource_, texture_id_, GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsWriteDiscard);

        if (err != cudaSuccess) {
            cleanup();
            throw std::runtime_error("Failed to register OpenGL texture with CUDA: " +
                                     std::string(cudaGetErrorString(err)));
        }

        is_registered_ = true;
    }

    void CudaGLInteropTexture::resize(int new_width, int new_height) {
        if (width_ != new_width || height_ != new_height) {
            init(new_width, new_height);
        }
    }

    void CudaGLInteropTexture::updateFromTensor(const torch::Tensor& image) {
        if (!is_registered_) {
            throw std::runtime_error("Texture not initialized");
        }

        // Ensure tensor is CUDA, float32, and [H, W, C] format
        TORCH_CHECK(image.is_cuda(), "Image must be on CUDA");
        TORCH_CHECK(image.dim() == 3, "Image must be [H, W, C]");
        TORCH_CHECK(image.size(2) == 3 || image.size(2) == 4,
                    "Image must have 3 or 4 channels");

        const int h = image.size(0);
        const int w = image.size(1);
        const int c = image.size(2);

        // Resize if needed
        resize(w, h);

        // Map CUDA resource
        cudaError_t err = cudaGraphicsMapResources(1, &cuda_resource_, 0);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to map CUDA resource: " +
                                     std::string(cudaGetErrorString(err)));
        }

        try {
            // Get CUDA array from mapped resource
            cudaArray_t cuda_array;
            err = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource_, 0, 0);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to get CUDA array: " +
                                         std::string(cudaGetErrorString(err)));
            }

            // Convert to RGBA uint8 if needed
            torch::Tensor rgba_image;
            if (c == 3) {
                // Add alpha channel
                rgba_image = torch::cat({image,
                                         torch::ones({h, w, 1}, image.options())},
                                        2);
            } else {
                rgba_image = image;
            }

            // Ensure proper format (uint8)
            if (rgba_image.dtype() != torch::kUInt8) {
                rgba_image = (rgba_image.clamp(0.0f, 1.0f) * 255.0f).to(torch::kUInt8);
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
                throw std::runtime_error("Failed to copy to CUDA array: " +
                                         std::string(cudaGetErrorString(err)));
            }

            // Synchronize to ensure copy is complete
            cudaDeviceSynchronize();

        } catch (...) {
            // Always unmap on error
            cudaGraphicsUnmapResources(1, &cuda_resource_, 0);
            throw;
        }

        // Unmap resource
        err = cudaGraphicsUnmapResources(1, &cuda_resource_, 0);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to unmap CUDA resource: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    void CudaGLInteropTexture::cleanup() {
        if (is_registered_ && cuda_resource_) {
            cudaGraphicsUnregisterResource(cuda_resource_);
            cuda_resource_ = nullptr;
            is_registered_ = false;
        }

        if (texture_id_ != 0) {
            glDeleteTextures(1, &texture_id_);
            texture_id_ = 0;
        }
    }

    // InteropFrameBuffer implementation
    InteropFrameBuffer::InteropFrameBuffer(bool use_interop)
        : FrameBuffer(),
          use_interop_(use_interop) {
        if (use_interop_) {
            try {
                interop_texture_.init(width, height);
            } catch (const std::exception& e) {
                std::cerr << "Failed to initialize CUDA-GL interop: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU copy mode" << std::endl;
                use_interop_ = false;
            }
        }
    }

    void InteropFrameBuffer::uploadFromCUDA(const torch::Tensor& cuda_image) {
        if (!use_interop_) {
            // Fallback to CPU copy
            auto cpu_image = cuda_image.to(torch::kCPU).contiguous();

            // Handle both [H, W, C] and [C, H, W] formats
            torch::Tensor formatted;
            if (cuda_image.size(-1) == 3 || cuda_image.size(-1) == 4) {
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
            return;
        }

        // Direct CUDA update
        try {
            interop_texture_.updateFromTensor(cuda_image);
        } catch (const std::exception& e) {
            std::cerr << "CUDA-GL interop update failed: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU copy" << std::endl;
            use_interop_ = false;
            uploadFromCUDA(cuda_image); // Retry with CPU fallback
        }
    }

    void InteropFrameBuffer::resize(int new_width, int new_height) {
        FrameBuffer::resize(new_width, new_height);
        if (use_interop_) {
            try {
                interop_texture_.resize(new_width, new_height);
            } catch (const std::exception& e) {
                std::cerr << "Failed to resize interop texture: " << e.what() << std::endl;
                use_interop_ = false;
            }
        }
    }

} // namespace gs

#else // CUDA_GL_INTEROP_ENABLED not defined

// Stub implementation when interop is not available
namespace gs {
    // Empty implementation - all functionality handled by preprocessor in header
}

#endif // CUDA_GL_INTEROP_ENABLED