#include "rendering/point_cloud_renderer.hpp"
#include "internal/resource_paths.hpp"
#include <iostream>

namespace gs {

    constexpr float PointCloudRenderer::cube_vertices_[];
    constexpr unsigned int PointCloudRenderer::cube_indices_[];

    PointCloudRenderer::PointCloudRenderer() = default;

    PointCloudRenderer::~PointCloudRenderer() {
        if (cube_vao_) glDeleteVertexArrays(1, &cube_vao_);
        if (cube_vbo_) glDeleteBuffers(1, &cube_vbo_);
        if (cube_ebo_) glDeleteBuffers(1, &cube_ebo_);
        if (instance_vbo_) glDeleteBuffers(1, &instance_vbo_);
        if (fbo_) glDeleteFramebuffers(1, &fbo_);
        if (color_texture_) glDeleteTextures(1, &color_texture_);
        if (depth_texture_) glDeleteTextures(1, &depth_texture_);
    }

    void PointCloudRenderer::initialize() {
        if (initialized_) return;

        // Create shader
        shader_ = std::make_unique<Shader>(
            (visualizer::getShaderPath("point_cloud.vert")).string().c_str(),
            (visualizer::getShaderPath("point_cloud.frag")).string().c_str(),
            false);

        createCubeGeometry();
        initialized_ = true;
    }

    void PointCloudRenderer::createCubeGeometry() {
        // Create VAO
        glGenVertexArrays(1, &cube_vao_);
        glBindVertexArray(cube_vao_);

        // Create VBO for cube vertices
        glGenBuffers(1, &cube_vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, cube_vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices_), cube_vertices_, GL_STATIC_DRAW);

        // Set vertex attributes for cube
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        // Create EBO for cube indices
        glGenBuffers(1, &cube_ebo_);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_ebo_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices_), cube_indices_, GL_STATIC_DRAW);

        // Create instance VBO for positions and colors
        glGenBuffers(1, &instance_vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, instance_vbo_);

        // Instance position attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribDivisor(1, 1);  // One per instance

        // Instance color attribute
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribDivisor(2, 1);  // One per instance

        glBindVertexArray(0);
    }

    torch::Tensor PointCloudRenderer::extractRGBFromSH(const torch::Tensor& shs) {
        // The SH coefficients are stored in a special format in gaussian splatting
        // shs shape: [N, K, 3] where K is number of SH coefficients
        
        // Get the DC component (first SH coefficient)
        torch::Tensor features_dc = shs.index({torch::indexing::Slice(), 0, torch::indexing::Slice()});
        
        // The SH coefficient for degree 0 (DC term) needs to be converted to color
        // Using the same formula as in the gaussian splatting renderer
        const float C0 = 0.28209479177387814f; // 1 / (2 * sqrt(pi))
        
        // Convert SH to RGB: color = sigmoid((features_dc * C0 + 0.5))
        torch::Tensor rgb = features_dc * C0 + 0.5f;
        
        // Apply sigmoid activation
        rgb = torch::sigmoid(rgb);
        
        return rgb;
    }

    void PointCloudRenderer::uploadPointData(const torch::Tensor& positions, const torch::Tensor& colors) {
        // Ensure tensors are on CPU and contiguous
        auto pos_cpu = positions.cpu().contiguous();
        auto col_cpu = colors.cpu().contiguous();

        // Interleave position and color data
        size_t num_points = positions.size(0);
        std::vector<float> instance_data(num_points * 6);

        auto pos_accessor = pos_cpu.accessor<float, 2>();
        auto col_accessor = col_cpu.accessor<float, 2>();

        for (size_t i = 0; i < num_points; ++i) {
            // Position
            instance_data[i * 6 + 0] = pos_accessor[i][0];
            instance_data[i * 6 + 1] = pos_accessor[i][1];
            instance_data[i * 6 + 2] = pos_accessor[i][2];
            // Color
            instance_data[i * 6 + 3] = col_accessor[i][0];
            instance_data[i * 6 + 4] = col_accessor[i][1];
            instance_data[i * 6 + 5] = col_accessor[i][2];
        }

        // Upload to GPU
        glBindBuffer(GL_ARRAY_BUFFER, instance_vbo_);
        glBufferData(GL_ARRAY_BUFFER, instance_data.size() * sizeof(float), 
                     instance_data.data(), GL_DYNAMIC_DRAW);

        current_point_count_ = num_points;
    }

    void PointCloudRenderer::render(const SplatData& splat_data,
                                   const glm::mat4& view,
                                   const glm::mat4& projection,
                                   float voxel_size,
                                   const glm::vec3& background_color) {
        if (!initialized_ || splat_data.size() == 0) return;

        // Get positions and SH coefficients
        torch::Tensor positions = splat_data.get_means();
        torch::Tensor shs = splat_data.get_shs();

        // Extract RGB from SH coefficients
        torch::Tensor colors = extractRGBFromSH(shs);

        // Upload data to GPU
        uploadPointData(positions, colors);

        // Setup rendering state
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClearColor(background_color.r, background_color.g, background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Bind shader and set uniforms
        shader_->bind();
        shader_->set_uniform("u_view", view);
        shader_->set_uniform("u_projection", projection);
        shader_->set_uniform("u_voxel_size", voxel_size);

        // Render instanced cubes
        glBindVertexArray(cube_vao_);
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0, 
                               static_cast<GLsizei>(current_point_count_));
        glBindVertexArray(0);

        shader_->unbind();
    }

} // namespace gs
