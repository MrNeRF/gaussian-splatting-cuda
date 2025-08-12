#pragma once

#include <ft2build.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <map>
#include <string>
#include FT_FREETYPE_H

namespace gs::gui {

    // Character structure to hold glyph data
    struct Character {
        GLuint textureID;   // ID handle of the glyph texture
        glm::ivec2 size;    // Size of glyph
        glm::ivec2 bearing; // Offset from baseline to left/top of glyph
        GLuint advance;     // Horizontal offset to advance to next glyph
    };

    class TextRenderer {
    public:
        TextRenderer(unsigned int width, unsigned int height);
        ~TextRenderer();

        bool LoadFont(const std::string& fontPath, unsigned int fontSize);
        void RenderText(const std::string& text, float x, float y, float scale,
                        const glm::vec3& color = glm::vec3(1.0f));
        void updateScreenSize(unsigned int width, unsigned int height) {
            screenWidth = width;
            screenHeight = height;
        }

    private:
        unsigned int screenWidth, screenHeight;
        GLuint VAO, VBO;
        GLuint shaderProgram;
        GLint uProjection, uTextColor;
        std::map<char, Character> characters;

        void initRenderData();
        bool compileShaders();
    };

} // namespace gs::gui