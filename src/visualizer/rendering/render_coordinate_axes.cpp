#include "rendering/render_coordinate_axes.hpp"
#include <iostream>

namespace gs {

    // Standard coordinate axes colors (RGB convention)
    const glm::vec3 RenderCoordinateAxes::X_AXIS_COLOR = glm::vec3(1.0f, 0.0f, 0.0f); // Red
    const glm::vec3 RenderCoordinateAxes::Y_AXIS_COLOR = glm::vec3(0.0f, 1.0f, 0.0f); // Green
    const glm::vec3 RenderCoordinateAxes::Z_AXIS_COLOR = glm::vec3(0.0f, 0.0f, 1.0f); // Blue

    RenderCoordinateAxes::RenderCoordinateAxes() : size_(2.0f),
                                                   line_width_(3.0f),
                                                   initialized_(false),
                                                   VAO_(0),
                                                   VBO_(0) {
        // All axes visible by default
        axis_visible_[0] = true; // X
        axis_visible_[1] = true; // Y
        axis_visible_[2] = true; // Z

        // Reserve space for 6 vertices (2 per axis: origin + endpoint)
        vertices_.reserve(6);
    }

    RenderCoordinateAxes::~RenderCoordinateAxes() {
        cleanup();
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
            std::string shader_path = std::string(PROJECT_ROOT_PATH) + "/src/visualizer/rendering/shaders/";
            shader_ = std::make_unique<Shader>(
                (shader_path + "coordinate_axes.vert").c_str(),
                (shader_path + "coordinate_axes.frag").c_str(),
                false); // Don't use shader's buffer management

            // Generate OpenGL objects
            glGenVertexArrays(1, &VAO_);
            glGenBuffers(1, &VBO_);

            initialized_ = true;

            // Initialize axes geometry
            createAxesGeometry();
            setupVertexData();

        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize CoordinateAxes: " << e.what() << std::endl;
            cleanup();
        }
    }

    void RenderCoordinateAxes::cleanup() {
        if (VAO_ != 0) {
            glDeleteVertexArrays(1, &VAO_);
            VAO_ = 0;
        }
        if (VBO_ != 0) {
            glDeleteBuffers(1, &VBO_);
            VBO_ = 0;
        }

        shader_.reset();
        initialized_ = false;
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
        if (!initialized_ || VAO_ == 0 || vertices_.empty())
            return;

        glBindVertexArray(VAO_);

        // Bind and upload vertex data
        glBindBuffer(GL_ARRAY_BUFFER, VBO_);
        glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(AxisVertex),
                     vertices_.data(), GL_DYNAMIC_DRAW);

        // Position attribute (location 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(AxisVertex),
                              (void*)offsetof(AxisVertex, position));
        glEnableVertexAttribArray(0);

        // Color attribute (location 1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(AxisVertex),
                              (void*)offsetof(AxisVertex, color));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }

    void RenderCoordinateAxes::render(const glm::mat4& view, const glm::mat4& projection) {
        if (!initialized_ || !shader_ || VAO_ == 0 || vertices_.empty())
            return;

        // Save current OpenGL state
        GLfloat current_line_width;
        glGetFloatv(GL_LINE_WIDTH, &current_line_width);
        GLboolean line_smooth_enabled = glIsEnabled(GL_LINE_SMOOTH);

        // Enable line rendering
        glEnable(GL_LINE_SMOOTH);
        glLineWidth(line_width_);

        // Bind shader and setup uniforms
        shader_->bind();

        try {
            // Set uniforms (axes are in world space, so no model transform needed)
            glm::mat4 mvp = projection * view;
            shader_->set_uniform("u_mvp", mvp);

            // Bind VAO and draw
            glBindVertexArray(VAO_);
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertices_.size()));
            glBindVertexArray(0);

        } catch (const std::exception& e) {
            std::cerr << "Error rendering coordinate axes: " << e.what() << std::endl;
        }

        shader_->unbind();

        // Restore OpenGL state
        glLineWidth(current_line_width);
        if (!line_smooth_enabled) {
            glDisable(GL_LINE_SMOOTH);
        }
    }

} // namespace gs