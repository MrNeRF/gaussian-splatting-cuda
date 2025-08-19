#include "bbox_renderer.hpp"
#include "shader_paths.hpp"

namespace gs::rendering {

    RenderBoundingBox::RenderBoundingBox() : color_(1.0f, 1.0f, 0.0f), // Yellow by default
                                             line_width_(2.0f),
                                             initialized_(false),
                                             VAO_(0),
                                             VBO_(0),
                                             EBO_(0) {
        // Initialize vertices vector with 8 vertices
        vertices_.resize(8);

        // Initialize indices vector with line indices
        indices_.assign(cube_line_indices_, cube_line_indices_ + 24);
    }

    RenderBoundingBox::~RenderBoundingBox() {
        cleanup();
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

            // Generate OpenGL objects
            glGenVertexArrays(1, &VAO_);
            glGenBuffers(1, &VBO_);
            glGenBuffers(1, &EBO_);

            initialized_ = true;

            // Initialize cube geometry with default bounds if not already set
            if (min_bounds_ == glm::vec3(0.0f) && max_bounds_ == glm::vec3(0.0f)) {
                setBounds(glm::vec3(-1.0f), glm::vec3(1.0f));
            } else {
                createCubeGeometry();
                setupVertexData();
            }

            // Check bindings *after* setup
            glBindVertexArray(VAO_);
            GLint vao_check, ebo_check;
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao_check);
            glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &ebo_check);
            glBindVertexArray(0);

        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize BoundingBox: " << e.what() << std::endl;
            cleanup();
        }
    }

    void RenderBoundingBox::cleanup() {
        if (VAO_ != 0) {
            glDeleteVertexArrays(1, &VAO_);
            VAO_ = 0;
        }
        if (VBO_ != 0) {
            glDeleteBuffers(1, &VBO_);
            VBO_ = 0;
        }
        if (EBO_ != 0) {
            glDeleteBuffers(1, &EBO_);
            EBO_ = 0;
        }

        shader_ = ManagedShader();
        initialized_ = false;
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
        if (!initialized_ || VAO_ == 0)
            return;

        glBindVertexArray(VAO_);

        // Bind and upload vertex data
        glBindBuffer(GL_ARRAY_BUFFER, VBO_);
        glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(glm::vec3),
                     vertices_.data(), GL_DYNAMIC_DRAW);

        // ✅ Bind and upload index data WHILE VAO is bound!
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned int),
                     indices_.data(), GL_STATIC_DRAW);

        // Vertex attribute setup
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(0);

        glBindVertexArray(0); // VAO now remembers VBO + EBO + attributes
    }

    void RenderBoundingBox::render(const glm::mat4& view, const glm::mat4& projection) {
        if (!initialized_ || !shader_.valid() || VAO_ == 0)
            return;

        // Save current OpenGL state
        GLfloat current_line_width;
        glGetFloatv(GL_LINE_WIDTH, &current_line_width);
        GLboolean line_smooth_enabled = glIsEnabled(GL_LINE_SMOOTH);

        // Enable line rendering
        glEnable(GL_LINE_SMOOTH);
        glLineWidth(line_width_);

        // Bind shader and setup uniforms
        ShaderScope s(shader_);

        try {

            auto box2World = world2BBox_.inv().toMat4();
            // Set uniforms
            glm::mat4 mvp = projection * view * box2World;

            s->set("u_mvp", mvp);
            s->set("u_color", color_);

            // Bind VAO and draw
            glBindVertexArray(VAO_);

            glDrawElements(GL_LINES, static_cast<GLsizei>(indices_.size()), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);

        } catch (const std::exception& e) {
            std::cerr << "Error rendering bounding box: " << e.what() << std::endl;
        }

        // Restore OpenGL state
        glLineWidth(current_line_width);
        if (!line_smooth_enabled) {
            glDisable(GL_LINE_SMOOTH);
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