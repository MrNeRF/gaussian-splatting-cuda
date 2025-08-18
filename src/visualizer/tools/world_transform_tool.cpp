#define GLM_ENABLE_EXPERIMENTAL

#include "tools/world_transform_tool.hpp"
#include "gui/ui_widgets.hpp"
#include "rendering/rendering.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <imgui.h>

namespace gs::visualizer {

    WorldTransformTool::WorldTransformTool()
        : ToolBase() {
        transform_ = std::make_shared<geometry::EuclideanTransform>();
    }

    WorldTransformTool::~WorldTransformTool() = default;

    bool WorldTransformTool::initialize(const ToolContext& ctx) {
        // Get rendering manager through the context
        auto* rendering_manager = ctx.getRenderingManager();
        if (!rendering_manager)
            return false;

        // Get rendering engine (it should be initialized by now)
        auto* engine = rendering_manager->getRenderingEngine();
        if (!engine)
            return false;

        try {
            // Create coordinate axes through the rendering engine
            coordinate_axes_ = engine->createCoordinateAxes();
            if (coordinate_axes_) {
                coordinate_axes_->setSize(axes_size_);
                // All axes visible by default
                coordinate_axes_->setAxisVisible(0, true);
                coordinate_axes_->setAxisVisible(1, true);
                coordinate_axes_->setAxisVisible(2, true);
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to create coordinate axes: " << e.what() << std::endl;
            return false;
        }

        return coordinate_axes_ != nullptr;
    }

    void WorldTransformTool::shutdown() {
        coordinate_axes_.reset();
    }

    void WorldTransformTool::update([[maybe_unused]] const ToolContext& ctx) {
        // Update axes if needed
        updateAxes();
    }

    void WorldTransformTool::render([[maybe_unused]] const ToolContext& ctx) {
        // Rendering is handled by the main rendering pipeline
        // The axes will be rendered if show_axes_ is true
    }

    void WorldTransformTool::renderUI(const gui::UIContext& ctx, bool* open) {
        if (!open || !*open)
            return;

        drawControls(ctx);
    }

    void WorldTransformTool::setTransform(const geometry::EuclideanTransform& transform) {
        *transform_ = transform;

        // Update UI values from transform
        // Extract Euler angles from rotation matrix
        glm::mat3 rot_mat = transform_->getRotationMat();
        glm::vec3 euler = glm::eulerAngles(glm::quat_cast(rot_mat));
        rotation_[0] = glm::degrees(euler.x);
        rotation_[1] = glm::degrees(euler.y);
        rotation_[2] = glm::degrees(euler.z);

        // Extract translation
        glm::vec3 trans = transform_->getTranslation();
        translation_[0] = trans.x;
        translation_[1] = trans.y;
        translation_[2] = trans.z;
    }

    bool WorldTransformTool::IsTrivialTrans() const {
        return transform_->isIdentity();
    }

    void WorldTransformTool::drawControls([[maybe_unused]] const gui::UIContext& ctx) {
        ImGui::Text("World Coordinate Transform");
        ImGui::Separator();

        // Show axes checkbox
        if (ImGui::Checkbox("Show Coordinate Axes", &show_axes_)) {
            if (coordinate_axes_) {
                // Update visibility
                coordinate_axes_->setAxisVisible(0, show_axes_);
                coordinate_axes_->setAxisVisible(1, show_axes_);
                coordinate_axes_->setAxisVisible(2, show_axes_);
            }
        }

        if (show_axes_) {
            ImGui::Indent();

            // Axes appearance
            if (ImGui::SliderFloat("Axes Size", &axes_size_, 0.5f, 10.0f)) {
                if (coordinate_axes_) {
                    coordinate_axes_->setSize(axes_size_);
                }
            }

            // Note: Line width is typically set in the renderer, not the axes object
            ImGui::SliderFloat("Line Width", &line_width_, 1.0f, 5.0f);

            ImGui::Unindent();
        }

        ImGui::Separator();
        ImGui::Text("Transform Parameters");

        bool transform_changed = false;

        // Rotation controls
        ImGui::Text("Rotation (degrees):");
        transform_changed |= ImGui::DragFloat3("##rotation", rotation_, 0.1f);

        // Translation controls
        ImGui::Text("Translation:");
        transform_changed |= ImGui::DragFloat3("##translation", translation_, 0.01f);

        if (transform_changed) {
            // Update transform from UI values
            glm::vec3 rot_rad(glm::radians(rotation_[0]),
                              glm::radians(rotation_[1]),
                              glm::radians(rotation_[2]));
            glm::vec3 trans(translation_[0], translation_[1], translation_[2]);

            *transform_ = geometry::EuclideanTransform(rot_rad.x, rot_rad.y, rot_rad.z,
                                                       trans.x, trans.y, trans.z);
        }

        // Reset button
        if (ImGui::Button("Reset Transform", ImVec2(-1, 0))) {
            resetTransform();
        }

        ImGui::Separator();
        drawTransformInfo();
    }

    void WorldTransformTool::drawTransformInfo() {
        ImGui::Text("Transform Info:");
        ImGui::Indent();

        if (IsTrivialTrans()) {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Identity Transform");
        } else {
            // Show rotation matrix
            glm::mat3 rot = transform_->getRotationMat();
            ImGui::Text("Rotation Matrix:");
            ImGui::Text("[%.3f, %.3f, %.3f]", rot[0][0], rot[1][0], rot[2][0]);
            ImGui::Text("[%.3f, %.3f, %.3f]", rot[0][1], rot[1][1], rot[2][1]);
            ImGui::Text("[%.3f, %.3f, %.3f]", rot[0][2], rot[1][2], rot[2][2]);

            // Show translation
            glm::vec3 trans = transform_->getTranslation();
            ImGui::Text("Translation: [%.3f, %.3f, %.3f]", trans.x, trans.y, trans.z);
        }

        ImGui::Unindent();
    }

    void WorldTransformTool::resetTransform() {
        *transform_ = geometry::EuclideanTransform();

        // Reset UI values
        rotation_[0] = rotation_[1] = rotation_[2] = 0.0f;
        translation_[0] = translation_[1] = translation_[2] = 0.0f;
    }

    void WorldTransformTool::updateAxes() {
        // Update axes properties if needed
        if (coordinate_axes_ && show_axes_) {
            // The axes visibility and size are already set
            // Additional updates can be done here if needed
        }
    }

} // namespace gs::visualizer