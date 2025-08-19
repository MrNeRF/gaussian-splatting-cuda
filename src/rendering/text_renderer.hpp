#pragma once

#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <map>
#include <string>

namespace gs::rendering {

    struct Character {
        GLuint textureID;
        glm::ivec2 size;
        glm::ivec2 bearing;
        GLuint advance;
    };

    class TextRenderer {
    public:
        TextRenderer(unsigned int width, unsigned int height);
        ~TextRenderer();

        bool LoadFont(const std::string& fontPath, unsigned int fontSize);
        void RenderText(const std::string& text, float x, float y, float scale,
                        const glm::vec3& color = glm::vec3(1.0f));
        void updateScreenSize(unsigned int width, unsigned int height);

    private:
        unsigned int screenWidth, screenHeight;
        VAO vao_;
        VBO vbo_;
        ManagedShader shader_;
        std::map<char, Character> characters;

        void initRenderData();
    };

} // namespace gs::rendering
