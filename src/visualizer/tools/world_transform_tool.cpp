#include "tools/world_transform_tool.hpp"
#include "core/events.hpp"

#include <imgui.h>

namespace gs::visualizer {

    WorldTransformTool::WorldTransformTool() : translation_(0.0f), angles_rad_(0.0f) {
        coordinate_axes_ = std::make_shared<RenderCoordinateAxes>();
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

    void WorldTransformTool::registerInputHandlers(InputHandler& handler) {
        //
    }

    void WorldTransformTool::onEnabledChanged(bool enabled) {
        if (!enabled) {
        }
    }

    void WorldTransformTool::setupEventHandlers() {
        using namespace events;
    }

    void WorldTransformTool::drawControls(const gs::gui::UIContext& ui_ctx) {
        if (!ImGui::CollapsingHeader("World Transform Box")) {
            return;
        }

        bool settings_changed = false;

        if (ImGui::Checkbox("Show Coord Axes", &show_axes_)) {
            settings_changed = true;
        }

        if (settings_changed) {
            // events::tools::CropBoxSettingsChanged{
            //     .show_box = show_crop_box_,
            //     .use_box = use_crop_box_}
            //     .emit();
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

            if (ImGui::Button("Reset to Default")) {
                coordinate_axes_->setSize(2);
                coordinate_axes_->setLineWidth(3);
            }

            // Rotation controls
            if (ImGui::TreeNode("Rotation")) {
                ImGui::Text("Ctrl+click for faster steps");
                ImGui::Text("Rotatate w.r.t world axes");

                ImGui::TreePop();
            }
        }

        // Translation controls
        if (ImGui::TreeNode("translation")) {
            ImGui::Text("Ctrl+click for faster steps");
            ImGui::Text("Trans w.r.t to world axes");

            bool world_trans_changed = false;
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

            if (world_trans_changed) {
                // bounding_box_->setBounds(
                //     glm::vec3(min_bounds[0], min_bounds[1], min_bounds[2]),
                //     glm::vec3(max_bounds[0], max_bounds[1], max_bounds[2]));
                //
                // events::ui::CropBoxChanged{
                //     .min_bounds = bounding_box_->getMinBounds(),
                //     .max_bounds = bounding_box_->getMaxBounds(),
                //     .enabled = use_crop_box_}
                //     .emit();
            }

            ImGui::Text("Translations: (%.3f, %.3f, %.3f)", translation_.x, translation_.y, translation_.z);
            ImGui::Text("Angles: (%.3f, %.3f, %.3f)", angles_rad_.x, angles_rad_.y, angles_rad_.z);

            ImGui::TreePop();
        }
    }

} // namespace gs::visualizer
