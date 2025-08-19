#include "axes_renderer.hpp"
#include "gl_state_guard.hpp"
#include "shader_paths.hpp"
#include <iostream>

namespace gs::rendering {

    // Standard coordinate axes colors (RGB convention)
    const glm::vec3 RenderCoordinateAxes::X_AXIS_COLOR = glm::vec3(1.0f, 0.0f, 0.0f); // Red
    const glm::vec3 RenderCoordinateAxes::Y_AXIS_COLOR = glm::vec3(0.0f, 1.0f, 0.0f); // Green
    const glm::vec3 RenderCoordinateAxes::Z_AXIS_COLOR = glm::vec3(0.0f, 0.0f, 1.0f); // Blue

    RenderCoordinateAxes::RenderCoordinateAxes() : size_(2.0f),
                                                   line_width_(3.0f),
                                                   initialized_(false) {
        // All axes visible by default
        axis_visible_[0] = true; // X
        axis_visible_[1] = true; // Y
        axis_visible_[2] = true; // Z

        // Reserve space for 6 vertices (2 per axis: origin + endpoint)
        vertices_.reserve(6);
    }

    void RenderCoordinateAxes::setSize(float size) {
        size_ = size;
        createAxesGeometry();

        if (isInitialized()) {
            setupVertexData();
        }
    }

    void RenderCoordinateAxes::setAxisVisible(int axis, bool visible) {
        if (axis >= 0 && axis < 3) {
            axis_visible_[axis] = visible;
            createAxesGeometry();

            if (isInitialized()) {
                setupVertexData();
            }
        }
    }

    bool RenderCoordinateAxes::isAxisVisible(int axis) const {
        if (axis >= 0 && axis < 3) {
            return axis_visible_[axis];
        }
        return false;
    }

    void RenderCoordinateAxes::init() {
        if (isInitialized())
            return;

        try {
            // Create shader for coordinate axes rendering
            auto result = load_shader("coordinate_axes", "coordinate_axes.vert", "coordinate_axes.frag", false);
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

            initialized_ = true;

            // Initialize axes geometry
            createAxesGeometry();
            setupVertexData();

        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize CoordinateAxes: " << e.what() << std::endl;
            initialized_ = false;
            throw;
        }
    }

    void RenderCoordinateAxes::createAxesGeometry() {
        vertices_.clear();

        // X-axis (Red)
        if (axis_visible_[0]) {
            vertices_.push_back({glm::vec3(0.0f, 0.0f, 0.0f), X_AXIS_COLOR});  // Origin
            vertices_.push_back({glm::vec3(size_, 0.0f, 0.0f), X_AXIS_COLOR}); // X endpoint
        }

        // Y-axis (Green)
        if (axis_visible_[1]) {
            vertices_.push_back({glm::vec3(0.0f, 0.0f, 0.0f), Y_AXIS_COLOR});  // Origin
            vertices_.push_back({glm::vec3(0.0f, size_, 0.0f), Y_AXIS_COLOR}); // Y endpoint
        }

        // Z-axis (Blue)
        if (axis_visible_[2]) {
            vertices_.push_back({glm::vec3(0.0f, 0.0f, 0.0f), Z_AXIS_COLOR});  // Origin
            vertices_.push_back({glm::vec3(0.0f, 0.0f, size_), Z_AXIS_COLOR}); // Z endpoint
        }
    }

    void RenderCoordinateAxes::setupVertexData() {
        if (!initialized_ || !vao_ || vertices_.empty())
            return;

        VAOBinder vao_bind(vao_);

        // Bind and upload vertex data
        BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);
        upload_buffer(GL_ARRAY_BUFFER, std::span(vertices_), GL_DYNAMIC_DRAW);

        // Position attribute (location 0)
        VertexAttribute position_attr{
            .index = 0,
            .size = 3,
            .type = GL_FLOAT,
            .normalized = GL_FALSE,
            .stride = sizeof(AxisVertex),
            .offset = (void*)offsetof(AxisVertex, position)};
        position_attr.apply();

        // Color attribute (location 1)
        VertexAttribute color_attr{
            .index = 1,
            .size = 3,
            .type = GL_FLOAT,
            .normalized = GL_FALSE,
            .stride = sizeof(AxisVertex),
            .offset = (void*)offsetof(AxisVertex, color)};
        color_attr.apply();
    }

    void RenderCoordinateAxes::render(const glm::mat4& view, const glm::mat4& projection) {
        if (!initialized_ || !shader_.valid() || !vao_ || vertices_.empty())
            return;

        // Use GLLineGuard for line width management
        GLLineGuard line_guard(line_width_);

        // Bind shader and setup uniforms
        ShaderScope s(shader_);

        try {
            // Set uniforms (axes are in world space, so no model transform needed)
            glm::mat4 mvp = projection * view;
            s->set("u_mvp", mvp);

            // Bind VAO and draw
            VAOBinder vao_bind(vao_);
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertices_.size()));

        } catch (const std::exception& e) {
            std::cerr << "Error rendering coordinate axes: " << e.what() << std::endl;
        }
    }

} // namespace gs::rendering