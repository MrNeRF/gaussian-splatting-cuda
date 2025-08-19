#include "text_renderer.hpp"
#include "gl_state_guard.hpp"
#include <ft2build.h>
#include <iostream>
#include FT_FREETYPE_H

namespace gs::rendering {

    TextRenderer::TextRenderer(unsigned int width, unsigned int height)
        : screenWidth(width),
          screenHeight(height) {
        initRenderData();
    }

    TextRenderer::~TextRenderer() {
        for (auto& pair : characters) {
            glDeleteTextures(1, &pair.second.textureID);
        }
    }

    bool TextRenderer::LoadFont(const std::string& fontPath, unsigned int fontSize) {
        for (auto& pair : characters) {
            glDeleteTextures(1, &pair.second.textureID);
        }
        characters.clear();

        FT_Library ft;
        if (FT_Init_FreeType(&ft)) {
            std::cerr << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;
            return false;
        }

        FT_Face face;
        if (FT_New_Face(ft, fontPath.c_str(), 0, &face)) {
            std::cerr << "ERROR::FREETYPE: Failed to load font from: " << fontPath << std::endl;
            FT_Done_FreeType(ft);
            return false;
        }

        FT_Set_Pixel_Sizes(face, 0, fontSize);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // Load only needed characters
        for (unsigned char c : {'X', 'Y', 'Z'}) {
            if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
                std::cerr << "ERROR::FREETYPE: Failed to load Glyph for character: " << c << std::endl;
                continue;
            }

            GLuint texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RED,
                face->glyph->bitmap.width,
                face->glyph->bitmap.rows,
                0,
                GL_RED,
                GL_UNSIGNED_BYTE,
                face->glyph->bitmap.buffer);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            Character character = {
                texture,
                glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
                glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
                static_cast<GLuint>(face->glyph->advance.x)};
            characters.insert(std::pair<char, Character>(c, character));
        }

        glBindTexture(GL_TEXTURE_2D, 0);
        FT_Done_Face(face);
        FT_Done_FreeType(ft);

        std::cout << "Successfully loaded font: " << fontPath << " with " << characters.size() << " characters" << std::endl;
        return true;
    }

    void TextRenderer::initRenderData() {
        // Load shader using shader manager
        auto result = load_shader("text_renderer", "text_renderer.vert", "text_renderer.frag", false);
        if (!result) {
            throw std::runtime_error(result.error().what());
        }
        shader_ = std::move(*result);

        // Create VAO and VBO
        auto vao_result = create_vao();
        if (!vao_result) {
            throw std::runtime_error(vao_result.error().what());
        }
        vao_ = std::move(*vao_result);

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            throw std::runtime_error(vbo_result.error().what());
        }
        vbo_ = std::move(*vbo_result);

        VAOBinder vao_bind(vao_);
        BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);

        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);

        VertexAttribute attr{
            .index = 0,
            .size = 4,
            .type = GL_FLOAT,
            .normalized = GL_FALSE,
            .stride = 4 * sizeof(float),
            .offset = nullptr};
        attr.apply();
    }

    void TextRenderer::updateScreenSize(unsigned int width, unsigned int height) {
        screenWidth = width;
        screenHeight = height;
    }

    void TextRenderer::RenderText(const std::string& text, float x, float y, float scale,
                                  const glm::vec3& color) {
        // Use RAII for OpenGL state management
        GLStateGuard state_guard;

        // Set up our rendering state
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBlendEquation(GL_FUNC_ADD);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glDisable(GL_SCISSOR_TEST);
        glDepthMask(GL_FALSE);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // Use shader with RAII scope
        ShaderScope s(shader_);

        // Set up orthographic projection
        glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(screenWidth),
                                          0.0f, static_cast<float>(screenHeight));
        s->set("projection", projection);
        s->set("textColor", color);

        // Ensure we're using texture unit 0
        glActiveTexture(GL_TEXTURE0);
        s->set("text", 0);

        // Bind our VAO
        VAOBinder vao_bind(vao_);

        // Iterate through all characters
        for (char c : text) {
            auto it = characters.find(c);
            if (it == characters.end())
                continue;

            Character ch = it->second;

            float xpos = x + ch.bearing.x * scale;
            float ypos = y - (ch.size.y - ch.bearing.y) * scale;

            float w = ch.size.x * scale;
            float h = ch.size.y * scale;

            // Update VBO for each character
            float vertices[6][4] = {
                {xpos, ypos + h, 0.0f, 0.0f},
                {xpos, ypos, 0.0f, 1.0f},
                {xpos + w, ypos, 1.0f, 1.0f},

                {xpos, ypos + h, 0.0f, 0.0f},
                {xpos + w, ypos, 1.0f, 1.0f},
                {xpos + w, ypos + h, 1.0f, 0.0f}};

            // Bind glyph texture
            glBindTexture(GL_TEXTURE_2D, ch.textureID);

            // Update content of VBO memory
            BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

            // Render quad
            glDrawArrays(GL_TRIANGLES, 0, 6);

            // Advance cursors for next glyph (advance is in 1/64 pixels)
            x += (ch.advance >> 6) * scale;
        }

        // State automatically restored by GLStateGuard destructor
    }

} // namespace gs::rendering
