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
        float wasdSpeed = 0.06f;

        float maxWasdSpeed = 1.0f;              // Maximum WASD speed
        float wasdSpeedChangePercentage = 1.0f; // 1% change per key press

        void increaseWasdSpeed() {
            wasdSpeed = std::min(wasdSpeed + maxWasdSpeed * wasdSpeedChangePercentage / 100, maxWasdSpeed);
        }

        void decreaseWasdSpeed() {
            wasdSpeed = std::max(wasdSpeed - maxWasdSpeed * wasdSpeedChangePercentage / 100, 0.001f); // Minimum speed to avoid zero
        }

        void setMaxWasdSpeed(float maxSpeed) {
            maxWasdSpeed = maxSpeed;
            wasdSpeed = std::min(wasdSpeed, maxWasdSpeed); // Clamp current speed if it exceeds new max
        }

        float getWasdSpeed() const {
            return wasdSpeed;
        }

        float getMaxWasdSpeed() const {
            return maxWasdSpeed;
        }

        void setWasdSpeedChangePercentage(float percentage) {
            wasdSpeedChangePercentage = std::max(0.01f, std::min(percentage, 1.0f)); // Clamp between 1% and 100%
        }

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
            t = U * t;
            R = U * R;
            prePos = pos;
        }

        void rotate_roll(float diff) {
            float ang_rad = diff * rotateRollSpeed;
            glm::mat3 rot_z = glm::mat3(
                glm::cos(ang_rad), -glm::sin(ang_rad), 0.0f,
                glm::sin(ang_rad), glm::cos(ang_rad), 0.0f,
                0.0f, 0.0f, 1.0f);

            R = R * rot_z;
        }

        void translate(const glm::vec2& pos) {
            glm::vec2 delta = pos - prePos;
            t -= (delta.x * translateSpeed) * R[0] + (delta.y * translateSpeed) * R[1];
            prePos = pos;
        }

        void zoom(float delta) {
            t += delta * zoomSpeed * R[2];
        }

        void advance_forward(float deltaTime) {

            t += R * glm::vec3(0, 0, 1) * deltaTime * wasdSpeed;
        }
        void advance_backward(float deltaTime) {
            t += R * glm::vec3(0, 0, -1) * deltaTime * wasdSpeed;
        }
        void advance_left(float deltaTime) {
            t += R * glm::vec3(-1, 0, 0) * deltaTime * wasdSpeed;
        }
        void advance_right(float deltaTime) {
            t += R * glm::vec3(1, 0, 0) * deltaTime * wasdSpeed;
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

    glm::mat4 getViewMatrix() const {
        // Convert R (3x3) and t (3x1) to a 4x4 view matrix
        // The view matrix transforms world coordinates to camera coordinates

        // In your system: camera.R is rotation, camera.t is translation
        // View matrix is the inverse of the camera transform

        glm::mat3 flip_yz = glm::mat3(
            1, 0, 0,
            0, -1, 0,
            0, 0, -1);

        glm::mat3 R_inv = glm::transpose(camera.R); // Inverse of rotation matrix
        glm::vec3 t_inv = -R_inv * camera.t;        // Inverse translation

        R_inv = flip_yz * R_inv;
        t_inv = flip_yz * t_inv;

        glm::mat4 view(1.0f);

        // Set rotation part (top-left 3x3)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                view[i][j] = R_inv[i][j];

        // Set translation part (last column)
        view[3][0] = t_inv.x;
        view[3][1] = t_inv.y;
        view[3][2] = t_inv.z;
        view[3][3] = 1.0f;

        return view;
    }

    glm::mat4 getProjectionMatrix(float fov_degrees = 60.0f, float near_plane = 0.1f, float far_plane = 1000.0f) const {
        // Create perspective projection matrix
        float aspect_ratio = static_cast<float>(windowSize.x) / static_cast<float>(windowSize.y);
        float fov_radians = glm::radians(fov_degrees);

        return glm::perspective(fov_radians, aspect_ratio, near_plane, far_plane);
    }
};
