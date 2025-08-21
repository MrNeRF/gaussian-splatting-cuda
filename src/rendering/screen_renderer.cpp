#include "screen_renderer.hpp"

#ifdef CUDA_GL_INTEROP_ENABLED
#include "cuda_gl_interop.hpp"
#endif

namespace gs::rendering {

    ScreenQuadRenderer::ScreenQuadRenderer(FrameBufferMode mode) {
        framebuffer = createFrameBuffer(mode);

        float quadVertices[] = {
            // positions   // texCoords
            -1.0f, 1.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f, 0.0f,

            -1.0f, 1.0f, 0.0f, 1.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f};

        auto vao_result = create_vao();
        if (!vao_result) {
            throw std::runtime_error(vao_result.error());
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            throw std::runtime_error(vbo_result.error());
        }
        quadVBO_ = std::move(*vbo_result);

        // Build VAO using VAOBuilder
        VAOBuilder builder(std::move(*vao_result));

        std::span<const float> vertices_span(quadVertices, sizeof(quadVertices) / sizeof(float));

        builder.attachVBO(quadVBO_, vertices_span, GL_STATIC_DRAW)
            .setAttribute({.index = 0,
                           .size = 2,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = 4 * sizeof(float),
                           .offset = nullptr,
                           .divisor = 0})
            .setAttribute({.index = 1,
                           .size = 2,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = 4 * sizeof(float),
                           .offset = (void*)(2 * sizeof(float)),
                           .divisor = 0});

        quadVAO_ = builder.build();
    }

    Result<void> ScreenQuadRenderer::render(std::shared_ptr<Shader> shader) const {
        if (!shader) {
            return std::unexpected("Shader is null");
        }

        shader->bind();

        VAOBinder vao_bind(quadVAO_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, getTextureID());

        try {
            shader->set_uniform("screenTexture", 0);
        } catch (const std::exception& e) {
            shader->unbind();
            return std::unexpected(std::format("Failed to set uniform: {}", e.what()));
        }

        glDrawArrays(GL_TRIANGLES, 0, 6);

        shader->unbind();
        return {};
    }

    Result<void> ScreenQuadRenderer::render(ManagedShader& shader) const {
        ShaderScope s(shader);

        VAOBinder vao_bind(quadVAO_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, getTextureID());

        if (auto result = shader.set("screenTexture", 0); !result) {
            return result;
        }

        glDrawArrays(GL_TRIANGLES, 0, 6);
        return {};
    }

    Result<void> ScreenQuadRenderer::uploadData(const unsigned char* image, int width_, int height_) {
        if (!framebuffer) {
            return std::unexpected("Framebuffer not initialized");
        }
        framebuffer->uploadImage(image, width_, height_);
        return {};
    }

    Result<void> ScreenQuadRenderer::uploadFromCUDA(const torch::Tensor& cuda_image, int width, int height) {
#ifdef CUDA_GL_INTEROP_ENABLED
        if (auto interop_fb = std::dynamic_pointer_cast<InteropFrameBuffer>(framebuffer)) {
            return interop_fb->uploadFromCUDA(cuda_image);
        }
#endif
        // Fallback to CPU upload
        auto cpu_image = cuda_image;
        if (cpu_image.dtype() != torch::kUInt8) {
            cpu_image = (cpu_image.clamp(0.0f, 1.0f) * 255.0f).to(torch::kUInt8);
        }
        cpu_image = cpu_image.to(torch::kCPU).contiguous();
        return uploadData(cpu_image.data_ptr<unsigned char>(), width, height);
    }

    bool ScreenQuadRenderer::isInteropEnabled() const {
#ifdef CUDA_GL_INTEROP_ENABLED
        return std::dynamic_pointer_cast<InteropFrameBuffer>(framebuffer) != nullptr;
#else
        return false;
#endif
    }

    GLuint ScreenQuadRenderer::getTextureID() const {
#ifdef CUDA_GL_INTEROP_ENABLED
        if (auto interop_fb = std::dynamic_pointer_cast<InteropFrameBuffer>(framebuffer)) {
            return interop_fb->getInteropTexture();
        }
#endif
        return framebuffer->getFrameTexture();
    }

} // namespace gs::rendering