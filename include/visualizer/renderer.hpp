#pragma once

#include "visualizer/framebuffer.hpp"
#include "visualizer/shader.hpp"
#include "visualizer/viewport.hpp"

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

    void render(std::shared_ptr<Shader> shader, const Viewport& viewport) const {

        shader->bind();

        glBindVertexArray(quadVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, framebuffer->getFrameTexture());

        shader->set_uniform("screenTexture", 0);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        shader->unbind();
    }

    void uploadData(const unsigned char* image, int width_, int height_) {
        framebuffer->uploadImage(image, width_, height_);
    }
};
