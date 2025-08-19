#pragma once

#include "core/splat_data.hpp"
#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>
#include <span>
#include <torch/torch.h>

namespace gs::rendering {

    class PointCloudRenderer {
    public:
        PointCloudRenderer() : initialized_(false),
                               current_point_count_(0) {
            std::cout << "[PointCloudRenderer] Constructor called at " << this << std::endl;
            std::cout << "[PointCloudRenderer] Stack trace would be helpful here" << std::endl;
        }
        ~PointCloudRenderer() = default;

        // Initialize OpenGL resources - now returns Result
        Result<void> initialize();

        // Render point cloud - now returns Result
        Result<void> render(const SplatData& splat_data,
                            const glm::mat4& view,
                            const glm::mat4& projection,
                            float voxel_size,
                            const glm::vec3& background_color);

        // Check if initialized
        bool isInitialized() const { return initialized_; }

    private:
        Result<void> createCubeGeometry();
        Result<void> uploadPointData(std::span<const float> positions, std::span<const float> colors);
        static torch::Tensor extractRGBFromSH(const torch::Tensor& shs);

        // OpenGL resources using RAII
        VAO cube_vao_;
        VBO cube_vbo_;
        EBO cube_ebo_;
        VBO instance_vbo_; // For positions and colors

        // Framebuffer resources using RAII
        FBO fbo_;
        Texture color_texture_;
        Texture depth_texture_;
        int fbo_width_ = 0;
        int fbo_height_ = 0;

        // Shaders
        ManagedShader shader_;

        // State
        bool initialized_ = false;
        size_t current_point_count_ = 0;

        // Cube vertices
        static constexpr float cube_vertices_[] = {
            // Front face
            -0.5f, -0.5f, 0.5f,
            0.5f, -0.5f, 0.5f,
            0.5f, 0.5f, 0.5f,
            -0.5f, 0.5f, 0.5f,
            // Back face
            -0.5f, -0.5f, -0.5f,
            0.5f, -0.5f, -0.5f,
            0.5f, 0.5f, -0.5f,
            -0.5f, 0.5f, -0.5f};

        // Triangle indices for solid cube (6 faces, 12 triangles, 36 indices)
        static constexpr unsigned int cube_indices_[] = {
            // Front face
            0, 1, 2, 2, 3, 0,
            // Back face
            4, 6, 5, 6, 4, 7,
            // Left face
            0, 3, 7, 7, 4, 0,
            // Right face
            1, 5, 6, 6, 2, 1,
            // Top face
            3, 2, 6, 6, 7, 3,
            // Bottom face
            0, 4, 5, 5, 1, 0};
    };

} // namespace gs::rendering