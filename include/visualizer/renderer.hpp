#pragma once

#include "config.h"
#include "visualizer/framebuffer.hpp"
#include "visualizer/shader.hpp"
#include "visualizer/viewport.hpp"

#ifdef CUDA_GL_INTEROP_ENABLED
#include "visualizer/cuda_gl_interop.hpp"
#endif

class ScreenQuadRenderer {

public:
    GLuint quadVAO;
    GLuint quadVBO;

    std::shared_ptr<FrameBuffer> framebuffer;

public:
    ScreenQuadRenderer() {

        framebuffer = std::make_shared<FrameBuffer>();

        float quadVertices[] = {
            // positions   // texCoords
            -1.0f, 1.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f, 0.0f,

            -1.0f, 1.0f, 0.0f, 1.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f};

        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);

        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

        glBindVertexArray(0);
    }

    virtual ~ScreenQuadRenderer() = default;

    virtual void render(std::shared_ptr<Shader> shader, const Viewport& viewport) const {

        shader->bind();

        glBindVertexArray(quadVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, getTextureID());

        shader->set_uniform("screenTexture", 0);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        shader->unbind();
    }

    virtual void uploadData(const unsigned char* image, int width_, int height_) {
        framebuffer->uploadImage(image, width_, height_);
    }

protected:
    virtual GLuint getTextureID() const {
        return framebuffer->getFrameTexture();
    }
};

#ifdef CUDA_GL_INTEROP_ENABLED

// Extended renderer with CUDA-OpenGL interop support
class ScreenQuadRendererInterop : public ScreenQuadRenderer {
private:
    std::shared_ptr<gs::InteropFrameBuffer> interop_framebuffer_;
    bool interop_enabled_;

public:
    explicit ScreenQuadRendererInterop(bool enable_interop = true)
        : ScreenQuadRenderer(),
          interop_enabled_(enable_interop) {

        if (interop_enabled_) {
            try {
                // Replace the base framebuffer with interop version
                interop_framebuffer_ = std::make_shared<gs::InteropFrameBuffer>(true);
                framebuffer = interop_framebuffer_;

                std::cout << "CUDA-OpenGL interop framebuffer initialized successfully" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to initialize interop framebuffer: " << e.what() << std::endl;
                std::cerr << "Falling back to standard framebuffer" << std::endl;
                interop_enabled_ = false;
                interop_framebuffer_ = nullptr;
                framebuffer = std::make_shared<FrameBuffer>();
            }
        }
    }

    // Upload directly from CUDA tensor
    void uploadFromCUDA(const torch::Tensor& cuda_image, int width, int height) {
        if (interop_framebuffer_ && interop_enabled_) {
            try {
                interop_framebuffer_->uploadFromCUDA(cuda_image);
            } catch (const std::exception& e) {
                std::cerr << "CUDA upload failed: " << e.what() << std::endl;
                // Fallback to CPU upload
                uploadData(cuda_image.to(torch::kCPU).to(torch::kUInt8).data_ptr<unsigned char>(),
                           width, height);
            }
        } else {
            // Fallback to CPU upload
            auto cpu_image = cuda_image;
            if (cpu_image.dtype() != torch::kUInt8) {
                cpu_image = (cpu_image.clamp(0.0f, 1.0f) * 255.0f).to(torch::kUInt8);
            }
            cpu_image = cpu_image.to(torch::kCPU).contiguous();
            uploadData(cpu_image.data_ptr<unsigned char>(), width, height);
        }
    }

    bool isInteropEnabled() const { return interop_enabled_; }

    void render(std::shared_ptr<Shader> shader, const Viewport& viewport) const override {
        shader->bind();

        glBindVertexArray(quadVAO);
        glActiveTexture(GL_TEXTURE0);

        // Use interop texture if available
        GLuint tex_id = interop_enabled_ && interop_framebuffer_ ? interop_framebuffer_->getInteropTexture() : framebuffer->getFrameTexture();

        glBindTexture(GL_TEXTURE_2D, tex_id);
        shader->set_uniform("screenTexture", 0);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        shader->unbind();
    }

protected:
    GLuint getTextureID() const override {
        if (interop_enabled_ && interop_framebuffer_) {
            return interop_framebuffer_->getInteropTexture();
        }
        return framebuffer->getFrameTexture();
    }
};

#else

// Stub for when interop is not available
using ScreenQuadRendererInterop = ScreenQuadRenderer;

#endif // CUDA_GL_INTEROP_ENABLED