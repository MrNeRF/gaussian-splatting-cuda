#pragma once

#include "cuda_gl_interop.hpp"
#include "framebuffer.hpp"
#include "gl_resources.hpp"
#include "shader.hpp"
#include "shader_manager.hpp"

#include <memory>

namespace gs::rendering {
    class ScreenQuadRenderer {
    protected:
        VAO quadVAO_;
        VBO quadVBO_;

    public:
        std::shared_ptr<FrameBuffer> framebuffer;

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

            auto vao_result = create_vao();
            if (!vao_result) {
                throw std::runtime_error(vao_result.error().what());
            }
            quadVAO_ = std::move(*vao_result);

            auto vbo_result = create_vbo();
            if (!vbo_result) {
                throw std::runtime_error(vbo_result.error().what());
            }
            quadVBO_ = std::move(*vbo_result);

            VAOBinder vao_bind(quadVAO_);
            BufferBinder<GL_ARRAY_BUFFER> vbo_bind(quadVBO_);
            upload_buffer(GL_ARRAY_BUFFER, quadVertices, 24, GL_STATIC_DRAW);

            VertexAttribute pos_attr{
                .index = 0,
                .size = 2,
                .type = GL_FLOAT,
                .normalized = GL_FALSE,
                .stride = 4 * sizeof(float),
                .offset = nullptr};
            pos_attr.apply();

            VertexAttribute tex_attr{
                .index = 1,
                .size = 2,
                .type = GL_FLOAT,
                .normalized = GL_FALSE,
                .stride = 4 * sizeof(float),
                .offset = (void*)(2 * sizeof(float))};
            tex_attr.apply();
        }

        virtual ~ScreenQuadRenderer() = default;

        virtual void render(std::shared_ptr<Shader> shader) const {
            shader->bind();

            VAOBinder vao_bind(quadVAO_);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, getTextureID());

            shader->set_uniform("screenTexture", 0);

            glDrawArrays(GL_TRIANGLES, 0, 6);

            shader->unbind();
        }

        // New overload for ManagedShader
        void render(ManagedShader& shader) const {
            ShaderScope s(shader);

            VAOBinder vao_bind(quadVAO_);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, getTextureID());

            shader.set("screenTexture", 0);

            glDrawArrays(GL_TRIANGLES, 0, 6);
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
        std::shared_ptr<InteropFrameBuffer> interop_framebuffer_;
        bool interop_enabled_;

    public:
        explicit ScreenQuadRendererInterop(bool enable_interop = true)
            : ScreenQuadRenderer(),
              interop_enabled_(enable_interop) {

            if (interop_enabled_) {
                try {
                    // Replace the base framebuffer with interop version
                    interop_framebuffer_ = std::make_shared<InteropFrameBuffer>(true);
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

        void render(std::shared_ptr<Shader> shader) const override {
            shader->bind();

            VAOBinder vao_bind(quadVAO_);
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
} // namespace gs::rendering