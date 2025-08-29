/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// clang-format off
// CRITICAL: GLAD must be included before GLFW to avoid OpenGL header conflicts
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include "core/logger.hpp"
#include <fstream>
#include <glm/glm.hpp>          // Add this include
#include <glm/gtc/type_ptr.hpp> // For glm::value_ptr
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gs::rendering {
    template <typename E>
    inline GLenum is_type_integral() {
        return GL_FALSE;
    }

    template <>
    inline GLenum is_type_integral<GLbyte>() {
        return GL_TRUE;
    }

    template <>
    inline GLenum is_type_integral<GLshort>() {
        return GL_TRUE;
    }

    template <>
    inline GLenum is_type_integral<GLint>() {
        return GL_TRUE;
    }

    template <>
    inline GLenum is_type_integral<GLubyte>() {
        return GL_TRUE;
    }

    template <>
    inline GLenum is_type_integral<GLushort>() {
        return GL_TRUE;
    }

    template <>
    inline GLenum is_type_integral<GLuint>() {
        return GL_TRUE;
    }

    template <typename E>
    inline GLenum get_type_enum() {
        LOG_ERROR("Error getting type enum: unsupported type");
        throw std::runtime_error("Error getting type enum: unsupported type");
    }

    template <>
    inline GLenum get_type_enum<GLbyte>() {
        return GL_BYTE;
    }

    template <>
    inline GLenum get_type_enum<GLshort>() {
        return GL_SHORT;
    }

    template <>
    inline GLenum get_type_enum<GLint>() {
        return GL_INT;
    }

    template <>
    inline GLenum get_type_enum<GLubyte>() {
        return GL_UNSIGNED_BYTE;
    }

    template <>
    inline GLenum get_type_enum<GLushort>() {
        return GL_UNSIGNED_SHORT;
    }

    template <>
    inline GLenum get_type_enum<GLuint>() {
        return GL_UNSIGNED_INT;
    }

    template <>
    inline GLenum get_type_enum<GLfloat>() {
        return GL_FLOAT;
    }

    class Shader {
    public:
        Shader(const char* vshader_path, const char* fshader_path, bool create_buffer = true) {
            LOG_TIMER_TRACE("Shader::Shader");
            LOG_DEBUG("Creating shader with vertex: {}", vshader_path);
            LOG_DEBUG("Creating shader with fragment: {}", fshader_path);

            GLint status;

            std::string vshader_source = readShaderSourceFromFile(vshader_path);
            std::string fshader_source = readShaderSourceFromFile(fshader_path);

            if (vshader_source.empty()) {
                LOG_ERROR("Vertex shader source is empty!");
                throw std::runtime_error("ERROR: Vertex shader source is empty!");
            }
            if (fshader_source.empty()) {
                LOG_ERROR("Fragment shader source is empty!");
                throw std::runtime_error("ERROR: Fragment shader source is empty!");
            }

            constexpr GLsizei MAX_INFO_LOG_LENGTH = 2000;
            GLsizei info_log_length;
            GLchar info_log[MAX_INFO_LOG_LENGTH];
            GLint compilation_status;
            auto check_comp_status = [&](GLuint shader) {
                glGetShaderiv(shader, GL_COMPILE_STATUS, &compilation_status);
                if (compilation_status == GL_TRUE)
                    return;
                glGetShaderInfoLog(shader, MAX_INFO_LOG_LENGTH, &info_log_length, info_log);
                std::string error_msg = "Shader compilation error:\n";
                error_msg += info_log;
                LOG_ERROR("Shader compilation error: {}", info_log);
                throw std::runtime_error(error_msg);
            };

            vshader = glCreateShader(GL_VERTEX_SHADER);
            const char* vshader_code = vshader_source.c_str();
            glShaderSource(vshader, 1, &vshader_code, nullptr);
            glCompileShader(vshader);
            check_comp_status(vshader);
            LOG_TRACE("Vertex shader compiled successfully");

            fshader = glCreateShader(GL_FRAGMENT_SHADER);
            const char* fshader_code = fshader_source.c_str();
            glShaderSource(fshader, 1, &fshader_code, nullptr);
            glCompileShader(fshader);
            check_comp_status(fshader);
            LOG_TRACE("Fragment shader compiled successfully");

            program = glCreateProgram();
            glAttachShader(program, vshader);
            glAttachShader(program, fshader);

            glLinkProgram(program);
            glGetProgramiv(program, GL_LINK_STATUS, &status);
            if (status != GL_TRUE) {
                glGetProgramInfoLog(program, MAX_INFO_LOG_LENGTH, nullptr, info_log);
                std::string error_msg = "Shader link error:\n";
                error_msg += info_log;
                LOG_ERROR("Shader link error: {}", info_log);
                throw std::runtime_error(error_msg);
            }
            LOG_DEBUG("Shader program linked successfully");

            if (create_buffer) {
                glGenBuffers(1, &index_buffer);
                glGenVertexArrays(1, &vertex_array);
                LOG_TRACE("Created index buffer and vertex array");
            }
        }

        ~Shader() {
            LOG_TRACE("Destroying shader program {}", program);
            for (auto [attrib, buffer] : attribute_buffers) {
                glDeleteBuffers(1, &buffer);
            }
            if (vertex_array != 0)
                glDeleteVertexArrays(1, &vertex_array);
            if (index_buffer != 0)
                glDeleteBuffers(1, &index_buffer);

            glDetachShader(program, fshader);
            glDetachShader(program, vshader);
            glDeleteProgram(program);
            glDeleteShader(fshader);
            glDeleteShader(vshader);
        }

        void bind(bool use_buffer = true) {
            if (use_buffer) {
                glBindVertexArray(vertex_array);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
            }
            glUseProgram(program);
        }

        void unbind(bool use_buffer = true) {
            if (use_buffer) {
                glBindVertexArray(0);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            }
            glUseProgram(0);
        }

        GLuint programID() const {
            return program;
        }

        void set_uniform(const std::string& name, const size_t& value) {
            GLint uni = uniform(name);
            glUniform1i(uni, value);
        }

        void set_uniform(const std::string& name, const int& value) {
            GLint uni = uniform(name);
            glUniform1i(uni, value);
        }

        void set_uniform(const std::string& name, const float& value) {
            GLint uni = uniform(name);
            glUniform1f(uni, value);
        }

        void set_uniform(const std::string& name, const glm::vec2& vector) {
            GLint uni = uniform(name);
            glUniform2fv(uni, 1, &vector[0]);
        }

        void set_uniform(const std::string& name, const glm::vec3& vector) {
            GLint uni = uniform(name);
            glUniform3fv(uni, 1, &vector[0]);
        }

        void set_uniform(const std::string& name, const glm::vec4& vector) {
            GLint uni = uniform(name);
            glUniform4fv(uni, 1, &vector[0]);
        }

        void set_uniform(const std::string& name, const glm::mat4& matrix) {
            GLint uni = uniform(name);
            glUniformMatrix4fv(uni, 1, GL_FALSE, &matrix[0][0]);
        }

        // texture
        void set_uniform(const std::string& name) {
            GLint uni = uniform(name);
            glUniform1i(uni, 0);
        }

        template <typename E, int N>
        void set_attribute(const std::string& name,
                           const std::vector<glm::vec<N, E, glm::defaultp>>& data) {
            LOG_TIMER_TRACE("Shader::set_attribute");
            GLint attrib = attribute(name);
            if (attribute_buffers.count(attrib) == 0) {
                GLuint buffer;
                glGenBuffers(1, &buffer);
                attribute_buffers[attrib] = buffer;
                LOG_TRACE("Created attribute buffer for '{}'", name);
            }
            GLuint buffer = attribute_buffers.at(attrib);
            glBindBuffer(GL_ARRAY_BUFFER, buffer);
            glBufferData(GL_ARRAY_BUFFER, sizeof(E) * N * data.size(), data.data(), GL_DYNAMIC_DRAW);
            glEnableVertexAttribArray(attrib);
            glVertexAttribPointer(attrib, N, get_type_enum<E>(), is_type_integral<E>(), 0, nullptr);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            LOG_TRACE("Set attribute '{}' with {} elements", name, data.size());
        }

        void set_indices(const std::vector<unsigned int>& indices) {
            LOG_TIMER_TRACE("Shader::set_indices");
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), &indices[0], GL_DYNAMIC_DRAW);
            LOG_TRACE("Set {} indices", indices.size());
        }

        void draw(GLenum mode, GLuint start, GLuint count) {
            glDrawArrays(mode, start, count);
        }

        void draw_indexed(GLenum mode, GLuint start, GLuint count) {
            glDrawElements(mode, count, GL_UNSIGNED_INT, (const void*)(start * sizeof(GLuint)));
        }

    private:
        std::string readShaderSourceFromFile(const std::string& filePath) {
            LOG_TIMER_TRACE("Shader::readShaderSourceFromFile");
            LOG_TRACE("Reading shader source from: {}", filePath);

            std::ifstream file(filePath);
            if (!file.is_open()) {
                LOG_ERROR("Failed to open shader file: {}", filePath);
                throw std::runtime_error("Failed to open shader file: " + filePath);
            }
            std::stringstream buffer;
            buffer << file.rdbuf();

            LOG_TRACE("Read {} bytes from shader file", buffer.str().size());
            return buffer.str();
        }

        GLint uniform(const std::string& name) {
            if (uniforms.count(name) == 0) {
                GLint location = glGetUniformLocation(program, name.c_str());
                if (location == -1) {
                    LOG_ERROR("Cannot find uniform '{}'", name);
                    throw std::runtime_error("Error: cannot find uniform '" + name + "'");
                }
                uniforms[name] = location;
                LOG_TRACE("Found uniform '{}' at location {}", name, location);
            }
            return uniforms.at(name);
        }

        GLint attribute(const std::string& name) {
            if (attributes.count(name) == 0) {
                GLint location = glGetAttribLocation(program, name.c_str());
                if (location == -1) {
                    LOG_ERROR("Error getting attribute location for '{}'", name);
                    throw std::runtime_error("Error getting attribute location for '" + name + "'");
                }
                attributes[name] = location;
                LOG_TRACE("Found attribute '{}' at location {}", name, location);
            }
            return attributes.at(name);
        }

        GLuint program;
        GLuint vshader;
        GLuint fshader;
        std::map<std::string, GLint> uniforms;
        std::map<std::string, GLint> attributes;
        std::map<GLint, GLuint> attribute_buffers;
        GLuint index_buffer = 0;
        GLuint vertex_array = 0;
    };
} // namespace gs::rendering