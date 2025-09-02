/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace gs::rendering {

    class CameraFrustumRenderer {
    public:
        CameraFrustumRenderer() = default;
        ~CameraFrustumRenderer() = default;

        Result<void> init();
        Result<void> render(const std::vector<std::shared_ptr<const Camera>>& cameras,
                            const glm::mat4& view,
                            const glm::mat4& projection,
                            float scale = 0.1f,
                            const glm::vec3& train_color = glm::vec3(0.0f, 1.0f, 0.0f),
                            const glm::vec3& eval_color = glm::vec3(1.0f, 0.0f, 0.0f));

        bool isInitialized() const { return initialized_; }

    private:
        Result<void> createGeometry();

        ManagedShader shader_;
        VAO vao_;
        VBO vbo_;
        EBO face_ebo_;
        EBO edge_ebo_;
        VBO instance_vbo_;

        size_t num_face_indices_ = 0;
        size_t num_edge_indices_ = 0;
        bool initialized_ = false;

        struct InstanceData {
            glm::mat4 transform;
            glm::vec3 color;
            float padding;
        };
    };

} // namespace gs::rendering
