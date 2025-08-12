#include "gui/text_renderer.hpp"

namespace gs::gui {

    static const char* textVertexShader = R"(#version 330 core
layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
out vec2 TexCoords;

uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
})";

    static const char* textFragmentShader = R"(#version 330 core
in vec2 TexCoords;
out vec4 color;

uniform sampler2D text;
uniform vec3 textColor;

void main()
{
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
    color = vec4(textColor, 1.0) * sampled;
})";

    TextRenderer::TextRenderer(unsigned int width, unsigned int height)
        : screenWidth(width),
          screenHeight(height),
          VAO(0),
          VBO(0),
          shaderProgram(0) {
        initRenderData();
    }

    TextRenderer::~TextRenderer() {
        if (VAO)
            glDeleteVertexArrays(1, &VAO);
        if (VBO)
            glDeleteBuffers(1, &VBO);
        if (shaderProgram)
            glDeleteProgram(shaderProgram);

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
                std::cerr << "ERROR::FREETYTPE: Failed to load Glyph for character: " << c << std::endl;
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

            GLenum err = glGetError();
            if (err != GL_NO_ERROR) {
                std::cerr << "OpenGL error after creating texture for character " << c << ": " << err << std::endl;
            }

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
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        compileShaders();
    }

    bool TextRenderer::compileShaders() {
        GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &textVertexShader, NULL);
        glCompileShader(vertex);

        GLint success;
        GLchar infoLog[512];
        glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vertex, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n"
                      << infoLog << std::endl;
            return false;
        }

        GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &textFragmentShader, NULL);
        glCompileShader(fragment);

        glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(fragment, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n"
                      << infoLog << std::endl;
            return false;
        }

        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertex);
        glAttachShader(shaderProgram, fragment);
        glLinkProgram(shaderProgram);

        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"
                      << infoLog << std::endl;
            return false;
        }

        glDeleteShader(vertex);
        glDeleteShader(fragment);

        uProjection = glGetUniformLocation(shaderProgram, "projection");
        uTextColor = glGetUniformLocation(shaderProgram, "textColor");

        return true;
    }

    void TextRenderer::RenderText(const std::string& text, float x, float y, float scale,
                                  const glm::vec3& color) {
        // Save comprehensive OpenGL state
        GLint current_program;
        glGetIntegerv(GL_CURRENT_PROGRAM, &current_program);
        GLint current_vao;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &current_vao);
        GLint current_active_texture;
        glGetIntegerv(GL_ACTIVE_TEXTURE, &current_active_texture);
        GLint current_texture_binding;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &current_texture_binding);
        GLboolean blend_enabled = glIsEnabled(GL_BLEND);
        GLboolean depth_enabled = glIsEnabled(GL_DEPTH_TEST);
        GLboolean cull_enabled = glIsEnabled(GL_CULL_FACE);
        GLboolean scissor_enabled = glIsEnabled(GL_SCISSOR_TEST);
        GLboolean stencil_enabled = glIsEnabled(GL_STENCIL_TEST);
        GLint blend_src, blend_dst;
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &blend_src);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &blend_dst);
        GLint blend_equation_rgb, blend_equation_alpha;
        glGetIntegerv(GL_BLEND_EQUATION_RGB, &blend_equation_rgb);
        glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &blend_equation_alpha);
        GLboolean depth_mask;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depth_mask);
        GLboolean color_mask[4];
        glGetBooleanv(GL_COLOR_WRITEMASK, color_mask);
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        GLint unpack_alignment;
        glGetIntegerv(GL_UNPACK_ALIGNMENT, &unpack_alignment);

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

        // Use our shader
        glUseProgram(shaderProgram);

        // Set up orthographic projection
        glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(screenWidth),
                                          0.0f, static_cast<float>(screenHeight));
        glUniformMatrix4fv(uProjection, 1, GL_FALSE, glm::value_ptr(projection));
        glUniform3f(uTextColor, color.x, color.y, color.z);

        // Ensure we're using texture unit 0
        glActiveTexture(GL_TEXTURE0);
        glUniform1i(glGetUniformLocation(shaderProgram, "text"), 0);

        // Bind our VAO
        glBindVertexArray(VAO);

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
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            // Render quad
            glDrawArrays(GL_TRIANGLES, 0, 6);

            // Advance cursors for next glyph (advance is in 1/64 pixels)
            x += (ch.advance >> 6) * scale;
        }

        // Restore ALL OpenGL state
        glUseProgram(current_program);
        glBindVertexArray(current_vao);
        glActiveTexture(current_active_texture);
        glBindTexture(GL_TEXTURE_2D, current_texture_binding);

        if (!blend_enabled)
            glDisable(GL_BLEND);
        else
            glEnable(GL_BLEND);

        if (depth_enabled)
            glEnable(GL_DEPTH_TEST);
        else
            glDisable(GL_DEPTH_TEST);

        if (cull_enabled)
            glEnable(GL_CULL_FACE);
        else
            glDisable(GL_CULL_FACE);

        if (scissor_enabled)
            glEnable(GL_SCISSOR_TEST);
        else
            glDisable(GL_SCISSOR_TEST);

        if (stencil_enabled)
            glEnable(GL_STENCIL_TEST);

        glBlendFunc(blend_src, blend_dst);
        glBlendEquationSeparate(blend_equation_rgb, blend_equation_alpha);
        glDepthMask(depth_mask);
        glColorMask(color_mask[0], color_mask[1], color_mask[2], color_mask[3]);
        glPixelStorei(GL_UNPACK_ALIGNMENT, unpack_alignment);
    }

} // namespace gs::gui