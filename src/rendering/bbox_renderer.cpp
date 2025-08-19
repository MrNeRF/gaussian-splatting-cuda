#include "bbox_renderer.hpp"
#include "gl_state_guard.hpp"
#include "shader_paths.hpp"

namespace gs::rendering {

    RenderBoundingBox::RenderBoundingBox() : color_(1.0f, 1.0f, 0.0f), // Yellow by default
                                             line_width_(2.0f),
                                             initialized_(false) {
        // Initialize vertices vector with 8 vertices
        vertices_.resize(8);

        // Initialize indices vector with line indices
        indices_.assign(cube_line_indices_, cube_line_indices_ + 24);
    }

    void RenderBoundingBox::setBounds(const glm::vec3& min, const glm::vec3& max) {
        // Call base class implementation
        BoundingBox::setBounds(min, max);
        createCubeGeometry();

        if (isInitialized()) {
            setupVertexData();
        }
    }

    void RenderBoundingBox::init() {
        if (isInitialized())
            return;

        try {
            // Create shader for bounding box rendering
            auto result = load_shader("bounding_box", "bounding_box.vert", "bounding_box.frag", false);
            if (!result) {
                throw std::runtime_error(result.error().what());
            }
            shader_ = std::move(*result);

            // Create OpenGL objects using RAII
            auto vao_result = create_vao();
            if (!vao_result) {
                throw std::runtime_error(vao_result.error().what());
            }
            vao_ = std::move(*vao_result);

            auto vbo_result = create_vbo();
            if (!vbo_result) {
                throw std::runtime_error(vbo_result.error().what());
            }
            vbo_ = std::move(*vbo_result);

            auto ebo_result = create_vbo(); // EBO is also a buffer
            if (!ebo_result) {
                throw std::runtime_error(ebo_result.error().what());
            }
            ebo_ = std::move(*ebo_result);

            initialized_ = true;

            // Initialize cube geometry with default bounds if not already set
            if (min_bounds_ == glm::vec3(0.0f) && max_bounds_ == glm::vec3(0.0f)) {
                setBounds(glm::vec3(-1.0f), glm::vec3(1.0f));
            } else {
                createCubeGeometry();
                setupVertexData();
            }

        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize BoundingBox: " << e.what() << std::endl;
            initialized_ = false;
            throw;
        }
    }

    void RenderBoundingBox::createCubeGeometry() {
        // Create 8 vertices of the bounding box cube
        vertices_[0] = glm::vec3(min_bounds_.x, min_bounds_.y, min_bounds_.z); // 0: min corner
        vertices_[1] = glm::vec3(max_bounds_.x, min_bounds_.y, min_bounds_.z); // 1: +x
        vertices_[2] = glm::vec3(max_bounds_.x, max_bounds_.y, min_bounds_.z); // 2: +x+y
        vertices_[3] = glm::vec3(min_bounds_.x, max_bounds_.y, min_bounds_.z); // 3: +y
        vertices_[4] = glm::vec3(min_bounds_.x, min_bounds_.y, max_bounds_.z); // 4: +z
        vertices_[5] = glm::vec3(max_bounds_.x, min_bounds_.y, max_bounds_.z); // 5: +x+z
        vertices_[6] = glm::vec3(max_bounds_.x, max_bounds_.y, max_bounds_.z); // 6: max corner
        vertices_[7] = glm::vec3(min_bounds_.x, max_bounds_.y, max_bounds_.z); // 7: +y+z
    }

    void RenderBoundingBox::setupVertexData() {
        if (!initialized_ || !vao_)
            return;

        VAOBinder vao_bind(vao_);

        // Bind and upload vertex data
        BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);
        upload_buffer(GL_ARRAY_BUFFER, std::span(vertices_), GL_DYNAMIC_DRAW);

        // Bind and upload index data
        BufferBinder<GL_ELEMENT_ARRAY_BUFFER> ebo_bind(ebo_);
        upload_buffer(GL_ELEMENT_ARRAY_BUFFER, std::span(indices_), GL_STATIC_DRAW);

        // Vertex attribute setup
        VertexAttribute position_attr{
            .index = 0,
            .size = 3,
            .type = GL_FLOAT,
            .normalized = GL_FALSE,
            .stride = sizeof(glm::vec3),
            .offset = nullptr};
        position_attr.apply();
    }

    void RenderBoundingBox::render(const glm::mat4& view, const glm::mat4& projection) {
        if (!initialized_ || !shader_.valid() || !vao_)
            return;

        // Use GLLineGuard for line width management
        GLLineGuard line_guard(line_width_);

        // Bind shader and setup uniforms
        ShaderScope s(shader_);

        try {
            auto box2World = world2BBox_.inv().toMat4();
            // Set uniforms
            glm::mat4 mvp = projection * view * box2World;

            s->set("u_mvp", mvp);
            s->set("u_color", color_);

            // Bind VAO and draw
            VAOBinder vao_bind(vao_);
            glDrawElements(GL_LINES, static_cast<GLsizei>(indices_.size()), GL_UNSIGNED_INT, 0);

        } catch (const std::exception& e) {
            std::cerr << "Error rendering bounding box: " << e.what() << std::endl;
        }
    }

    const unsigned int RenderBoundingBox::cube_line_indices_[24] = {
        // Bottom face edges
        0, 1, 1, 2, 2, 3, 3, 0,
        // Top face edges
        4, 5, 5, 6, 6, 7, 7, 4,
        // Vertical edges
        0, 4, 1, 5, 2, 6, 3, 7};

} // namespace gs::rendering