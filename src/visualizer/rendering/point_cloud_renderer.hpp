#pragma once

#include "core/splat_data.hpp"
#include "rendering/shader.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <torch/torch.h>

namespace gs {

    class PointCloudRenderer {
    public:
        PointCloudRenderer();
        ~PointCloudRenderer();

        // Initialize OpenGL resources
        void initialize();

        // Render point cloud
        void render(const SplatData& splat_data,
                    const glm::mat4& view,
                    const glm::mat4& projection,
                    float voxel_size,
                    const glm::vec3& background_color);

        // Check if initialized
        bool isInitialized() const { return initialized_; }

    private:
        void createCubeGeometry();
        void uploadPointData(const torch::Tensor& positions, const torch::Tensor& colors);
        static torch::Tensor extractRGBFromSH(const torch::Tensor& shs);

        // OpenGL resources
        GLuint cube_vao_ = 0;
        GLuint cube_vbo_ = 0;
        GLuint cube_ebo_ = 0;
        GLuint instance_vbo_ = 0; // For positions and colors

        // Framebuffer resources
        GLuint fbo_ = 0;
        GLuint color_texture_ = 0;
        GLuint depth_texture_ = 0;
        int fbo_width_ = 0;
        int fbo_height_ = 0;

        // Shaders
        std::unique_ptr<Shader> shader_;

        // State
        bool initialized_ = false;
        size_t current_point_count_ = 0;

        // Cube vertices and indices
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

} // namespace gs