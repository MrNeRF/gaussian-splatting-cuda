#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>
#include <iostream>

class Viewport {

    class CameraMotion {
    public:
        glm::vec2 prePos;

        float zoomSpeed = 1.0f;
        float rotateSpeed = 0.001f;
        float rotateCenterSpeed = 0.002f;
        float rotateRollSpeed = 0.01f;
        float translateSpeed = 0.001f;

        glm::mat3 R = glm::mat3(1.0f);
        glm::vec3 t = glm::vec3(0.0f);

        CameraMotion() = default;

        void rotate(const glm::vec2& pos) {
            glm::vec2 delta = pos - prePos;
            float y = -delta.x * rotateSpeed;
            float p = +delta.y * rotateSpeed;
            glm::mat3 Ry = glm::mat3(glm::rotate(glm::mat4(1.0f), y, R[1]));
            glm::mat3 Rp = glm::mat3(glm::rotate(glm::mat4(1.0f), p, R[0]));
            R = Rp * Ry * R;
            prePos = pos;
        }

        void rotate_around_center(const glm::vec2& pos) {
            glm::vec2 delta = pos - prePos;
            float y = +delta.x * rotateCenterSpeed;
            float p = -delta.y * rotateCenterSpeed;
            glm::mat3 Ry = glm::mat3(glm::rotate(glm::mat4(1.0f), y, R[1]));
            glm::mat3 Rp = glm::mat3(glm::rotate(glm::mat4(1.0f), p, R[0]));
            auto U = Rp * Ry;
            t = U*t;
            R = U*R;
            prePos = pos;
        }

        void rotate_roll(float diff) {
            float ang_rad = diff*rotateRollSpeed;
            glm::mat3 rot_z = glm::mat3(
                glm::cos(ang_rad), -glm::sin(ang_rad), 0.0f,
                glm::sin(ang_rad),  glm::cos(ang_rad), 0.0f,
                0.0f,               0.0f,              1.0f
            );

            R = R*rot_z;

        }

        void translate(const glm::vec2& pos) {
            glm::vec2 delta = pos - prePos;
            t -= (delta.x * translateSpeed) * R[0] + (delta.y * translateSpeed) * R[1];
            prePos = pos;
        }

        void zoom(float delta) {
            t += delta * zoomSpeed * R[2];
        }

        void initScreenPos(const glm::vec2& pos) {
            prePos = pos;
        }
    };

public:
    glm::ivec2 windowSize;
    glm::ivec2 frameBufferSize;
    CameraMotion camera;

    Viewport(size_t width = 1280, size_t height = 720) {

        windowSize = glm::ivec2(width, height);

        camera = CameraMotion();
    }

    void setViewMatrix(const glm::mat3& R, const glm::vec3& t) {
        camera.R = R;
        camera.t = t;
    }

    glm::mat3 getRotationMatrix() const {
        return camera.R;
    }

    glm::vec3 getTranslation() const {
        return camera.t;
    }
};
