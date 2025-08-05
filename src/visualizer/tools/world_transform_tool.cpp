#include "tools/world_transform_tool.hpp"
#include "core/events.hpp"

#include <imgui.h>

namespace gs::visualizer {

    WorldTransformTool::WorldTransformTool() : translation_(0.0f), angles_deg_(0.0f) {
        coordinate_axes_ = std::make_shared<RenderCoordinateAxes>();
        world_to_user = std::make_shared<geometry::EuclideanTransform>();
        setupEventHandlers();
    }

    WorldTransformTool::~WorldTransformTool() = default;

    bool WorldTransformTool::initialize(const ToolContext& ctx) {
        return true;
    }

    void WorldTransformTool::shutdown() {
        // Cleanup handled by destructors
    }

    void WorldTransformTool::update(const ToolContext& ctx) {
    }

    void WorldTransformTool::render(const ToolContext& ctx) {
        // Rendering is handled by the rendering manager based on our state
        // This method could be used for tool-specific overlays if needed
    }

    void WorldTransformTool::renderUI(const gs::gui::UIContext& ui_ctx, bool* p_open) {
        if (!isEnabled()) {
            return;
        }

        drawControls(ui_ctx);
    }

    void WorldTransformTool::onEnabledChanged(bool enabled) {
        if (!enabled) {
        }
    }

    void WorldTransformTool::setupEventHandlers() {
        using namespace events;
        // Listen for axes settings changes
        tools::AxesSettingsChanged::when([this](const auto& e) {
            show_axes_ = e.show_axes;
        });
    }

    float wrapAngle(float angle) {
        angle = fmod(angle, 360.0f);
        if (angle < 0)
            angle += 360.0f;
        return angle;
    }

    void WorldTransformTool::drawControls(const gs::gui::UIContext& ui_ctx) {
        if (!ImGui::CollapsingHeader("World Transform")) {
            return;
        }

        bool settings_changed = false;

        if (ImGui::Checkbox("Show Coord Axes", &show_axes_)) {
            settings_changed = true;
        }

        if (settings_changed) {

            events::tools::AxesSettingsChanged{
                .show_axes = show_axes_}
                .emit();
        }
        if (ImGui::Button("Reset to Default")) {
            coordinate_axes_->setSize(2);
            coordinate_axes_->setLineWidth(3);
            translation_ = glm::vec3{0};
            angles_deg_ = glm::vec3{0};
        }

        if (show_axes_) {
            float available_width = ImGui::GetContentRegionAvail().x;
            float button_width = 120.0f;
            float slider_width = available_width - button_width - ImGui::GetStyle().ItemSpacing.x;

            ImGui::SetNextItemWidth(slider_width);
            if (ImGui::SliderFloat("Axes Line Width", &line_width_, 0.5f, 10.0f)) {
                coordinate_axes_->setLineWidth(line_width_);
            }
            if (ImGui::SliderFloat("Axes Size", &axes_size_, 0.5f, 10.0f)) {
                coordinate_axes_->setSize(axes_size_);
            }
        }
        bool world_rot_changed = false;
        bool world_trans_changed = false;

        // Rotation controls
        if (ImGui::TreeNode("Rotation")) {
            ImGui::Text("Ctrl+click for faster steps");
            ImGui::Text("Rotatate w.r.t world axes (Deg) ");
            const float step = 1.0f;
            const float step_fast = 5.0f;

            // Trans X
            ImGui::Text("RotX:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(110);
            world_rot_changed |= ImGui::InputFloat("RotX", &angles_deg_[0], step, step_fast, "%.3f");
            angles_deg_[0] = wrapAngle(angles_deg_[0]);

            ImGui::Text("RotY:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(110);
            world_rot_changed |= ImGui::InputFloat("RotY", &angles_deg_[1], step, step_fast, "%.3f");
            angles_deg_[1] = wrapAngle(angles_deg_[1]);

            ImGui::Text("RotZ:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(110);
            world_rot_changed |= ImGui::InputFloat("RotZ", &angles_deg_[2], step, step_fast, "%.3f");
            angles_deg_[2] = wrapAngle(angles_deg_[2]);

            ImGui::TreePop();
        }

        // Translation controls
        if (ImGui::TreeNode("translation")) {
            ImGui::Text("Ctrl+click for faster steps");
            ImGui::Text("Trans w.r.t to world axes");

            const float min_range = -8.0f;
            const float max_range = 8.0f;
            const float step = 0.01f;
            const float step_fast = 0.1f;

            // Trans X
            ImGui::Text("X:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(110);
            world_trans_changed |= ImGui::InputFloat("TransX", &translation_[0], step, step_fast, "%.3f");
            translation_[0] = std::clamp(translation_[0], min_range, max_range);

            // Trans Y
            ImGui::Text("Y:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(110);
            world_trans_changed |= ImGui::InputFloat("TransY", &translation_[1], step, step_fast, "%.3f");
            translation_[1] = std::clamp(translation_[1], min_range, max_range);

            // Trans Z
            ImGui::Text("Z:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(110);
            world_trans_changed |= ImGui::InputFloat("TransZ", &translation_[2], step, step_fast, "%.3f");
            translation_[2] = std::clamp(translation_[2], min_range, max_range);

            ImGui::Text("Translations: (%.3f, %.3f, %.3f)", translation_.x, translation_.y, translation_.z);
            ImGui::Text("Angles: (%.3f, %.3f, %.3f)", angles_deg_.x, angles_deg_.y, angles_deg_.z);

            ImGui::TreePop();
        }
    }

    bool WorldTransformTool::IsTrivialTrans() const {
        return translation_ == glm::vec3(0.0f) && angles_deg_ == glm::vec3(0.0f);
    }

    [[nodiscard]] std::shared_ptr<const geometry::EuclideanTransform> WorldTransformTool::GetTransform() {
        *world_to_user = {glm::radians(angles_deg_), translation_};
        return world_to_user;
    }

} // namespace gs::visualizer
