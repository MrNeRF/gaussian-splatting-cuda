#include "core/bounding_box.hpp"
// ============================================================================
    // BoundingBox.cpp - Implementation
    // ============================================================================
namespace gs {
    const unsigned int BoundingBox::cube_line_indices_[24] = {
        // Bottom face edges
        0, 1, 1, 2, 2, 3, 3, 0,
        // Top face edges
        4, 5, 5, 6, 6, 7, 7, 4,
        // Vertical edges
        0, 4, 1, 5, 2, 6, 3, 7
    };

    BoundingBox::BoundingBox()
        : min_bounds_(-1.0f, -1.0f, -1.0f)
        , max_bounds_(1.0f, 1.0f, 1.0f)
        , transform_(1.0f)
        , color_(1.0f, 1.0f, 0.0f) // Yellow by default
        , line_width_(2.0f)
        , visible_(false)
        , initialized_(false)
        , VAO_(0), VBO_(0), EBO_(0)
    {
        // Initialize vertices vector with 8 vertices
        vertices_.resize(8);

        // Initialize indices vector with line indices
        indices_.assign(cube_line_indices_, cube_line_indices_ + 24);
    }

    BoundingBox::~BoundingBox() {
        cleanup();
    }

    void BoundingBox::cleanup() {
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
    }

    void BoundingBox::init() {
        if (initialized_) return;

        try {
            // Create shader for bounding box rendering
            std::string shader_path = std::string(PROJECT_ROOT_PATH) + "/include/visualizer/shaders/";
            shader_ = std::make_unique<Shader>(
                (shader_path + "bounding_box.vert").c_str(),
                (shader_path + "bounding_box.frag").c_str(),
                false); // Don't use shader's buffer management

            // Generate OpenGL objects
            glGenVertexArrays(1, &VAO_);
            glGenBuffers(1, &VBO_);
            glGenBuffers(1, &EBO_);

            // Initialize cube geometry
            createCubeGeometry();
            setupVertexData();

            initialized_ = true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize BoundingBox: " << e.what() << std::endl;
            cleanup();
        }
    }

    void BoundingBox::setBounds(const glm::vec3& min, const glm::vec3& max) {
        // Validate bounds
        if (min.x > max.x || min.y > max.y || min.z > max.z) {
            std::cerr << "Warning: Invalid bounding box bounds (min > max)" << std::endl;
            return;
        }

        min_bounds_ = min;
        max_bounds_ = max;
        createCubeGeometry();

        if (initialized_) {
            setupVertexData();
        }
    }

    void BoundingBox::setTransform(const glm::mat4& transform) {
        transform_ = transform;
    }

    void BoundingBox::createCubeGeometry() {
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

    void BoundingBox::setupVertexData() {
        if (!initialized_ || VAO_ == 0) return;

        // Bind VAO
        glBindVertexArray(VAO_);

        // Upload vertex data
        glBindBuffer(GL_ARRAY_BUFFER, VBO_);
        glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(glm::vec3),
                     vertices_.data(), GL_DYNAMIC_DRAW);

        // Upload index data
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned int),
                     indices_.data(), GL_STATIC_DRAW);

        // Set vertex attribute (position)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(0);

        // Unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void BoundingBox::render(const glm::mat4& view, const glm::mat4& projection) {
        if (!visible_ || !initialized_ || !shader_ || VAO_ == 0) return;

        // Save current OpenGL state
        GLfloat current_line_width;
        glGetFloatv(GL_LINE_WIDTH, &current_line_width);
        GLboolean line_smooth_enabled = glIsEnabled(GL_LINE_SMOOTH);

        // Enable line rendering
        glEnable(GL_LINE_SMOOTH);
        glLineWidth(line_width_);

        // Disable depth writing but keep depth testing for proper ordering
        glDepthMask(GL_FALSE);

        // Bind shader and setup uniforms
        shader_->bind();

        try {
            // Set uniforms
            glm::mat4 model = transform_;
            glm::mat4 mvp = projection * view * model;

            shader_->set_uniform("u_mvp", mvp);
            shader_->set_uniform("u_color", color_);

            // Bind VAO and draw
            glBindVertexArray(VAO_);
            glDrawElements(GL_LINES, static_cast<GLsizei>(indices_.size()), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);

        } catch (const std::exception& e) {
            std::cerr << "Error rendering bounding box: " << e.what() << std::endl;
        }

        shader_->unbind();

        // Restore OpenGL state
        glDepthMask(GL_TRUE);
        glLineWidth(current_line_width);
        if (!line_smooth_enabled) {
            glDisable(GL_LINE_SMOOTH);
        }
    }

    void BoundingBox::updateFromModel(const torch::Tensor& positions) {
        if (positions.numel() == 0) {
            std::cerr << "Warning: Empty positions tensor for bounding box update" << std::endl;
            return;
        }

        try {
            // Convert to CPU if needed and ensure contiguous
            auto pos_cpu = positions.cpu().contiguous();

            // Validate tensor dimensions
            if (pos_cpu.dim() != 2 || pos_cpu.size(1) != 3) {
                std::cerr << "Error: Expected positions tensor of shape [N, 3], got "
                          << pos_cpu.sizes() << std::endl;
                return;
            }

            auto pos_accessor = pos_cpu.accessor<float, 2>();

            // Find min/max bounds
            glm::vec3 min_pos(FLT_MAX);
            glm::vec3 max_pos(-FLT_MAX);

            for (int64_t i = 0; i < pos_cpu.size(0); ++i) {
                glm::vec3 pos(pos_accessor[i][0], pos_accessor[i][1], pos_accessor[i][2]);

                // Check for invalid values
                if (std::isfinite(pos.x) && std::isfinite(pos.y) && std::isfinite(pos.z)) {
                    min_pos = glm::min(min_pos, pos);
                    max_pos = glm::max(max_pos, pos);
                }
            }

            // Ensure we found valid bounds
            if (min_pos.x == FLT_MAX || max_pos.x == -FLT_MAX) {
                std::cerr << "Warning: No valid positions found for bounding box" << std::endl;
                return;
            }

            setBounds(min_pos, max_pos);

        } catch (const std::exception& e) {
            std::cerr << "Error updating bounding box from model: " << e.what() << std::endl;
        }
    }

    void BoundingBox::autoFit(const torch::Tensor& positions, float padding) {
        updateFromModel(positions);

        // Add padding
        glm::vec3 size = max_bounds_ - min_bounds_;
        glm::vec3 center = (min_bounds_ + max_bounds_) * 0.5f;

        // Ensure minimum size to avoid degenerate bounding boxes
        const float min_size = 0.001f;
        for (int i = 0; i < 3; ++i) {
            if (size[i] < min_size) {
                size[i] = min_size;
            }
        }

        glm::vec3 pad = size * padding;
        glm::vec3 half_size = size * 0.5f + pad;

        setBounds(center - half_size, center + half_size);
    }
}