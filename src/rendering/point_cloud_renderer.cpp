#include "point_cloud_renderer.hpp"
#include "shader_paths.hpp"
#include <vector>

namespace gs::rendering {

    Result<void> PointCloudRenderer::initialize() {
        std::cout << "[PointCloudRenderer] initialize() called on instance " << this << std::endl;

        if (initialized_) {
            std::cout << "[PointCloudRenderer] WARNING: Already initialized!" << std::endl;
            return {};
        }

        // Create shader
        auto result = load_shader("point_cloud", "point_cloud.vert", "point_cloud.frag", false);
        if (!result) {
            return std::unexpected(result.error().what());
        }
        shader_ = std::move(*result);

        if (auto geom_result = createCubeGeometry(); !geom_result) {
            return geom_result;
        }

        initialized_ = true;
        return {};
    }

    Result<void> PointCloudRenderer::createCubeGeometry() {
        // Create all resources first
        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected(vao_result.error());
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            return std::unexpected(vbo_result.error());
        }
        cube_vbo_ = std::move(*vbo_result);

        auto ebo_result = create_vbo(); // EBO is also a buffer
        if (!ebo_result) {
            return std::unexpected(ebo_result.error());
        }
        cube_ebo_ = std::move(*ebo_result);

        auto instance_result = create_vbo();
        if (!instance_result) {
            return std::unexpected(instance_result.error());
        }
        instance_vbo_ = std::move(*instance_result);

        // Build VAO using VAOBuilder
        VAOBuilder builder(std::move(*vao_result));

        // Setup cube geometry
        std::span<const float> vertices_span(cube_vertices_,
                                             sizeof(cube_vertices_) / sizeof(float));
        builder.attachVBO(cube_vbo_, vertices_span, GL_STATIC_DRAW)
            .setAttribute({.index = 0,
                           .size = 3,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = 3 * sizeof(float),
                           .offset = nullptr,
                           .divisor = 0});

        // Setup instance attributes (structure only, data comes later)
        builder.attachVBO(instance_vbo_) // Attach without data
            .setAttribute({
                .index = 1,
                .size = 3,
                .type = GL_FLOAT,
                .normalized = GL_FALSE,
                .stride = 6 * sizeof(float),
                .offset = nullptr,
                .divisor = 1 // Instance attribute
            })
            .setAttribute({
                .index = 2,
                .size = 3,
                .type = GL_FLOAT,
                .normalized = GL_FALSE,
                .stride = 6 * sizeof(float),
                .offset = (void*)(3 * sizeof(float)),
                .divisor = 1 // Instance attribute
            });

        // Attach EBO - stays bound to VAO
        std::span<const unsigned int> indices_span(cube_indices_,
                                                   sizeof(cube_indices_) / sizeof(unsigned int));
        builder.attachEBO(cube_ebo_, indices_span, GL_STATIC_DRAW);

        // Build and store the VAO
        cube_vao_ = builder.build();

        return {};
    }

    torch::Tensor PointCloudRenderer::extractRGBFromSH(const torch::Tensor& shs) {
        const float SH_C0 = 0.28209479177387814f;
        torch::Tensor features_dc = shs.index({torch::indexing::Slice(), 0, torch::indexing::Slice()});
        torch::Tensor colors = features_dc * SH_C0 + 0.5f;
        return colors.clamp(0.0f, 1.0f);
    }

    Result<void> PointCloudRenderer::uploadPointData(std::span<const float> positions, std::span<const float> colors) {
        // Using span, we can calculate the number of points
        size_t num_points = positions.size() / 3;

        // Validate sizes
        if (positions.size() != num_points * 3 || colors.size() != num_points * 3) {
            return std::unexpected("Invalid position or color data size");
        }

        // Interleave position and color data
        std::vector<float> instance_data(num_points * 6);

        for (size_t i = 0; i < num_points; ++i) {
            // Position
            instance_data[i * 6 + 0] = positions[i * 3 + 0];
            instance_data[i * 6 + 1] = positions[i * 3 + 1];
            instance_data[i * 6 + 2] = positions[i * 3 + 2];
            // Color - ensure values are in [0, 1] range
            instance_data[i * 6 + 3] = std::clamp(colors[i * 3 + 0], 0.0f, 1.0f);
            instance_data[i * 6 + 4] = std::clamp(colors[i * 3 + 1], 0.0f, 1.0f);
            instance_data[i * 6 + 5] = std::clamp(colors[i * 3 + 2], 0.0f, 1.0f);
        }

        // Upload to GPU
        BufferBinder<GL_ARRAY_BUFFER> bind(instance_vbo_);
        upload_buffer(GL_ARRAY_BUFFER, std::span(instance_data), GL_DYNAMIC_DRAW);

        current_point_count_ = num_points;
        return {};
    }

    Result<void> PointCloudRenderer::render(const SplatData& splat_data,
                                            const glm::mat4& view,
                                            const glm::mat4& projection,
                                            float voxel_size,
                                            const glm::vec3& background_color) {
        if (!initialized_) {
            return std::unexpected("Renderer not initialized");
        }

        if (splat_data.size() == 0) {
            return {}; // Nothing to render
        }

        // Get positions and SH coefficients
        torch::Tensor positions = splat_data.get_means();
        torch::Tensor shs = splat_data.get_shs();

        // Extract RGB colors from SH coefficients
        torch::Tensor colors = extractRGBFromSH(shs);

        // Ensure tensors are on CPU and contiguous
        auto pos_cpu = positions.cpu().contiguous();
        auto col_cpu = colors.cpu().contiguous();

        // Validate tensor dimensions
        if (pos_cpu.numel() == 0 || col_cpu.numel() == 0) {
            return std::unexpected("Empty tensor after CPU conversion");
        }

        if (pos_cpu.numel() % 3 != 0) {
            return std::unexpected("Invalid position tensor dimensions");
        }

        if (col_cpu.numel() % 3 != 0) {
            return std::unexpected("Invalid color tensor dimensions");
        }

        // Create spans for the data
        if (!pos_cpu.data_ptr<float>() || !col_cpu.data_ptr<float>()) {
            return std::unexpected("Null tensor data pointer");
        }

        std::span<const float> pos_span(pos_cpu.data_ptr<float>(), pos_cpu.numel());
        std::span<const float> col_span(col_cpu.data_ptr<float>(), col_cpu.numel());

        // Upload data to GPU
        if (auto result = uploadPointData(pos_span, col_span); !result) {
            return result;
        }

        // Validate instance count
        if (current_point_count_ > 10000000) { // 10 million sanity check
            return std::unexpected("Instance count exceeds reasonable limit");
        }

        // Setup rendering state
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClearColor(background_color.r, background_color.g, background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Bind shader and set uniforms
        ShaderScope s(shader_);
        if (auto result = s->set("u_view", view); !result) {
            return result;
        }
        if (auto result = s->set("u_projection", projection); !result) {
            return result;
        }
        if (auto result = s->set("u_voxel_size", voxel_size); !result) {
            return result;
        }

        // Validate VAO
        if (!cube_vao_ || cube_vao_.get() == 0) {
            return std::unexpected("Invalid cube VAO");
        }

        // Render instanced cubes
        if (current_point_count_ == 0) {
            return {};
        }

        VAOBinder vao_bind(cube_vao_);
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0,
                                static_cast<GLsizei>(current_point_count_));

        // Check for OpenGL errors
        GLenum gl_error = glGetError();
        if (gl_error != GL_NO_ERROR) {
            return std::unexpected(std::format("OpenGL error after draw call: 0x{:x}", gl_error));
        }

        return {};
    }

} // namespace gs::rendering