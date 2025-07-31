#include "tools/crop_box_tool.hpp"
#include "core/events.hpp"
#include "gui/ui_context.hpp"
#include "gui/ui_widgets.hpp"

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <cmath>
#include <imgui.h>

namespace gs::visualizer {

    CropBoxTool::CropBoxTool() {
        bounding_box_ = std::make_shared<RenderBoundingBox>();
        setupEventHandlers();
    }

    CropBoxTool::~CropBoxTool() = default;

    bool CropBoxTool::initialize(const ToolContext& ctx) {
        // Bounding box OpenGL resources are initialized lazily on first render
        return true;
    }

    void CropBoxTool::shutdown() {
        // Cleanup handled by destructors
    }

    void CropBoxTool::update(const ToolContext& ctx) {
        // Nothing to update per frame for crop box
    }

    void CropBoxTool::render(const ToolContext& ctx) {
        // Rendering is handled by the rendering manager based on our state
        // This method could be used for tool-specific overlays if needed
    }

    void CropBoxTool::renderUI(const gs::gui::UIContext& ui_ctx, bool* p_open) {
        if (!isEnabled()) {
            return;
        }

        drawControls(ui_ctx);
    }

    void CropBoxTool::registerInputHandlers(InputHandler& handler) {
        // Clear any existing handlers
        unregisterInputHandlers(handler);

        // Register mouse button handler for clicking on crop box handles
        handler_ids_.push_back(
            handler.addMouseButtonHandler(
                [this](const InputHandler::MouseButtonEvent& event) {
                    if (!isEnabled() || !shouldShowBox())
                        return false;

                    if (event.action == GLFW_PRESS && event.button == GLFW_MOUSE_BUTTON_LEFT) {
                        // Check if clicking on any handle
                        if (isMouseOverHandle(event.position)) {
                            startDragging(event.position);
                            return true; // Consume event
                        }
                    } else if (event.action == GLFW_RELEASE && event.button == GLFW_MOUSE_BUTTON_LEFT) {
                        if (is_dragging_) {
                            stopDragging();
                            return true; // Consume event
                        }
                    }
                    return false;
                },
                InputPriority::Tools));

        // Register mouse move handler for dragging
        handler_ids_.push_back(
            handler.addMouseMoveHandler(
                [this](const InputHandler::MouseMoveEvent& event) {
                    if (!isEnabled() || !is_dragging_)
                        return false;

                    updateDragging(event.position);
                    return true; // Consume event while dragging
                },
                InputPriority::Tools));

        // Register key handler for quick toggles
        handler_ids_.push_back(
            handler.addKeyHandler(
                [this](const InputHandler::KeyEvent& event) {
                    if (!isEnabled() || event.action != GLFW_PRESS)
                        return false;

                    // Ctrl+B to toggle crop box visibility
                    if (event.key == GLFW_KEY_B && (event.mods & GLFW_MOD_CONTROL)) {
                        show_crop_box_ = !show_crop_box_;
                        events::tools::CropBoxSettingsChanged{
                            .show_box = show_crop_box_,
                            .use_box = use_crop_box_}
                            .emit();
                        return true;
                    }

                    return false;
                },
                InputPriority::Tools));
    }

    void CropBoxTool::onEnabledChanged(bool enabled) {
        if (!enabled) {
            // Optionally disable crop box when tool is disabled
            show_crop_box_ = false;
            use_crop_box_ = false;
            is_dragging_ = false;

            events::tools::CropBoxSettingsChanged{
                .show_box = false,
                .use_box = false}
                .emit();
        }
    }

    void CropBoxTool::setupEventHandlers() {
        using namespace events;

        // Listen for crop box settings changes
        tools::CropBoxSettingsChanged::when([this](const auto& e) {
            show_crop_box_ = e.show_box;
            use_crop_box_ = e.use_box;
        });
    }

    bool CropBoxTool::isMouseOverHandle(const glm::dvec2& mouse_pos) const {
        // For now, just return false
        return false;
    }

    void CropBoxTool::startDragging(const glm::dvec2& mouse_pos) {
        is_dragging_ = true;
        drag_start_pos_ = mouse_pos;
        drag_start_box_min_ = bounding_box_->getMinBounds();
        drag_start_box_max_ = bounding_box_->getMaxBounds();

        // Determine which handle is being dragged
        // This would involve projecting handles to screen space
        current_handle_ = DragHandle::Center; // Simplified
    }

    void CropBoxTool::updateDragging(const glm::dvec2& mouse_pos) {
        if (!is_dragging_)
            return;

        glm::dvec2 delta = mouse_pos - drag_start_pos_;

        // Convert screen delta to world space delta
        // This is simplified - real implementation would use proper unprojection
        float scale = 0.01f;
        glm::vec3 world_delta(delta.x * scale, -delta.y * scale, 0.0f);

        // Update bounds based on which handle is being dragged
        switch (current_handle_) {
        case DragHandle::Center:
            bounding_box_->setBounds(
                drag_start_box_min_ + world_delta,
                drag_start_box_max_ + world_delta);
            break;
        // Add cases for other handles...
        default:
            break;
        }

        events::ui::CropBoxChanged{
            .min_bounds = bounding_box_->getMinBounds(),
            .max_bounds = bounding_box_->getMaxBounds(),
            .enabled = use_crop_box_}
            .emit();
    }

    void CropBoxTool::stopDragging() {
        is_dragging_ = false;
        current_handle_ = DragHandle::None;
    }

    void CropBoxTool::drawControls(const gs::gui::UIContext& ui_ctx) {
        if (!ImGui::CollapsingHeader("Crop Box")) {
            return;
        }

        bool settings_changed = false;

        if (ImGui::Checkbox("Show Crop Box", &show_crop_box_)) {
            settings_changed = true;
        }

        if (ImGui::Checkbox("Use Crop Box", &use_crop_box_)) {
            settings_changed = true;
        }

        if (settings_changed) {
            events::tools::CropBoxSettingsChanged{
                .show_box = show_crop_box_,
                .use_box = use_crop_box_}
                .emit();
        }

        if (show_crop_box_ && bounding_box_->isInitialized()) {
            // Appearance controls
            if (ImGui::ColorEdit3("Box Color", bbox_color_)) {
                bounding_box_->setColor(glm::vec3(bbox_color_[0], bbox_color_[1], bbox_color_[2]));
            }

            float available_width = ImGui::GetContentRegionAvail().x;
            float button_width = 120.0f;
            float slider_width = available_width - button_width - ImGui::GetStyle().ItemSpacing.x;

            ImGui::SetNextItemWidth(slider_width);
            if (ImGui::SliderFloat("Line Width", &line_width_, 0.5f, 10.0f)) {
                bounding_box_->setLineWidth(line_width_);
            }

            if (ImGui::Button("Reset to Default")) {
                bounding_box_->setBounds(glm::vec3(-1.0f), glm::vec3(1.0f));
                bounding_box_->setworld2BBox(glm::mat4(1.0));
            }

            // Rotation controls
            if (ImGui::TreeNode("Rotation")) {
                ImGui::Text("Ctrl+click for faster steps");
                ImGui::Text("Rotation around crop box axes:");

                const float rotation_step = 1.0f;
                const float rotation_step_fast = 15.0f;

                float step = ImGui::GetIO().KeyCtrl ? rotation_step_fast : rotation_step;
                float repeat_rate = 0.05f;

                float diff_x = 0, diff_y = 0, diff_z = 0;

                // X-axis rotation
                ImGui::Text("X-axis:");
                ImGui::SameLine();
                ImGui::Text("RotX");

                if (ImGui::ArrowButton("##RotX_Up", ImGuiDir_Up)) {
                    diff_x = step;
                    rotate_timer_x_ = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_x_ += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_x_ >= repeat_rate) {
                        diff_x = step;
                        rotate_timer_x_ = 0.0f;
                    }
                }

                ImGui::SameLine();
                if (ImGui::ArrowButton("##RotX_Down", ImGuiDir_Down)) {
                    diff_x = -step;
                    rotate_timer_x_ = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_x_ += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_x_ >= repeat_rate) {
                        diff_x = -step;
                        rotate_timer_x_ = 0.0f;
                    }
                }

                // Y-axis rotation
                ImGui::Text("Y-axis:");
                ImGui::SameLine();
                ImGui::Text("RotY");

                if (ImGui::ArrowButton("##RotY_Up", ImGuiDir_Up)) {
                    diff_y = step;
                    rotate_timer_y_ = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_y_ += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_y_ >= repeat_rate) {
                        diff_y = step;
                        rotate_timer_y_ = 0.0f;
                    }
                }

                ImGui::SameLine();
                if (ImGui::ArrowButton("##RotY_Down", ImGuiDir_Down)) {
                    diff_y = -step;
                    rotate_timer_y_ = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_y_ += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_y_ >= repeat_rate) {
                        diff_y = -step;
                        rotate_timer_y_ = 0.0f;
                    }
                }

                // Z-axis rotation
                ImGui::Text("Z-axis:");
                ImGui::SameLine();
                ImGui::Text("RotZ");

                if (ImGui::ArrowButton("##RotZ_Up", ImGuiDir_Up)) {
                    diff_z = step;
                    rotate_timer_z_ = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_z_ += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_z_ >= repeat_rate) {
                        diff_z = step;
                        rotate_timer_z_ = 0.0f;
                    }
                }

                ImGui::SameLine();
                if (ImGui::ArrowButton("##RotZ_Down", ImGuiDir_Down)) {
                    diff_z = -step;
                    rotate_timer_z_ = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_z_ += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_z_ >= repeat_rate) {
                        diff_z = -step;
                        rotate_timer_z_ = 0.0f;
                    }
                }

                if (diff_x != 0 || diff_y != 0 || diff_z != 0) {
                    updateRotationMatrix(diff_x, diff_y, diff_z);
                }

                ImGui::TreePop();
            }

            // Bounds controls
            if (ImGui::TreeNode("Bounds")) {
                glm::vec3 current_min = bounding_box_->getMinBounds();
                glm::vec3 current_max = bounding_box_->getMaxBounds();

                float min_bounds[3] = {current_min.x, current_min.y, current_min.z};
                float max_bounds[3] = {current_max.x, current_max.y, current_max.z};

                bool bounds_changed = false;

                const float min_range = -8.0f;
                const float max_range = 8.0f;
                const float bound_step = 0.01f;
                const float bound_step_fast = 0.1f;

                ImGui::Text("Ctrl+click for faster steps");
                ImGui::Text("Min Bounds:");

                // Min bounds
                ImGui::Text("X:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(110);
                bounds_changed |= ImGui::InputFloat("##MinX", &min_bounds[0], bound_step, bound_step_fast, "%.3f");
                min_bounds[0] = std::clamp(min_bounds[0], min_range, max_range);
                min_bounds[0] = std::min(min_bounds[0], max_bounds[0]);

                ImGui::Text("Y:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(110);
                bounds_changed |= ImGui::InputFloat("##MinY", &min_bounds[1], bound_step, bound_step_fast, "%.3f");
                min_bounds[1] = std::clamp(min_bounds[1], min_range, max_range);
                min_bounds[1] = std::min(min_bounds[1], max_bounds[1]);

                ImGui::Text("Z:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(110);
                bounds_changed |= ImGui::InputFloat("##MinZ", &min_bounds[2], bound_step, bound_step_fast, "%.3f");
                min_bounds[2] = std::clamp(min_bounds[2], min_range, max_range);
                min_bounds[2] = std::min(min_bounds[2], max_bounds[2]);

                ImGui::Separator();
                ImGui::Text("Max Bounds:");

                // Max bounds
                ImGui::Text("X:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(110);
                bounds_changed |= ImGui::InputFloat("##MaxX", &max_bounds[0], bound_step, bound_step_fast, "%.3f");
                max_bounds[0] = std::clamp(max_bounds[0], min_range, max_range);
                max_bounds[0] = std::max(max_bounds[0], min_bounds[0]);

                ImGui::Text("Y:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(110);
                bounds_changed |= ImGui::InputFloat("##MaxY", &max_bounds[1], bound_step, bound_step_fast, "%.3f");
                max_bounds[1] = std::clamp(max_bounds[1], min_range, max_range);
                max_bounds[1] = std::max(max_bounds[1], min_bounds[1]);

                ImGui::Text("Z:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(110);
                bounds_changed |= ImGui::InputFloat("##MaxZ", &max_bounds[2], bound_step, bound_step_fast, "%.3f");
                max_bounds[2] = std::clamp(max_bounds[2], min_range, max_range);
                max_bounds[2] = std::max(max_bounds[2], min_bounds[2]);

                if (bounds_changed) {
                    bounding_box_->setBounds(
                        glm::vec3(min_bounds[0], min_bounds[1], min_bounds[2]),
                        glm::vec3(max_bounds[0], max_bounds[1], max_bounds[2]));

                    events::ui::CropBoxChanged{
                        .min_bounds = bounding_box_->getMinBounds(),
                        .max_bounds = bounding_box_->getMaxBounds(),
                        .enabled = use_crop_box_}
                        .emit();
                }

                // Display info
                glm::vec3 center = bounding_box_->getCenter();
                glm::vec3 size = bounding_box_->getSize();

                ImGui::Text("Center: (%.3f, %.3f, %.3f)", center.x, center.y, center.z);
                ImGui::Text("Size: (%.3f, %.3f, %.3f)", size.x, size.y, size.z);

                ImGui::TreePop();
            }
        }
    }

    void CropBoxTool::updateRotationMatrix(float delta_rot_x, float delta_rot_y, float delta_rot_z) {
        if (!bounding_box_)
            return;

        float rad_x = glm::radians(delta_rot_x);
        float rad_y = glm::radians(delta_rot_y);
        float rad_z = glm::radians(delta_rot_z);

        glm::mat4 rot_x = glm::mat4(1.0f);
        rot_x[1][1] = cos(rad_x);
        rot_x[1][2] = sin(rad_x);
        rot_x[2][1] = -sin(rad_x);
        rot_x[2][2] = cos(rad_x);

        glm::mat4 rot_y = glm::mat4(1.0f);
        rot_y[0][0] = cos(rad_y);
        rot_y[0][2] = -sin(rad_y);
        rot_y[2][0] = sin(rad_y);
        rot_y[2][2] = cos(rad_y);

        glm::mat4 rot_z = glm::mat4(1.0f);
        rot_z[0][0] = cos(rad_z);
        rot_z[0][1] = sin(rad_z);
        rot_z[1][0] = -sin(rad_z);
        rot_z[1][1] = cos(rad_z);

        glm::mat4 combined_rotation = rot_x * rot_y * rot_z;

        glm::vec3 center = bounding_box_->getLocalCenter();

        glm::mat4 translate_to_origin = glm::mat4(1.0f);
        translate_to_origin[3][0] = -center.x;
        translate_to_origin[3][1] = -center.y;
        translate_to_origin[3][2] = -center.z;

        glm::mat4 translate_back = glm::mat4(1.0f);
        translate_back[3][0] = center.x;
        translate_back[3][1] = center.y;
        translate_back[3][2] = center.z;

        glm::mat4 rotation_transform = translate_back * combined_rotation * translate_to_origin;
        glm::mat4 curr_world2bbox = bounding_box_->getworld2BBox();
        glm::mat4 final_transform = rotation_transform * curr_world2bbox;
        final_transform = orthonormalizeRotation(final_transform);

        bounding_box_->setworld2BBox(final_transform);
    }

    glm::mat4 CropBoxTool::orthonormalizeRotation(const glm::mat4& matrix) {
        glm::vec3 x = glm::vec3(matrix[0]);
        glm::vec3 y = glm::vec3(matrix[1]);
        glm::vec3 z = glm::vec3(matrix[2]);

        x = glm::normalize(x);
        y = glm::normalize(y - x * glm::dot(x, y));
        z = glm::normalize(glm::cross(x, y));

        glm::mat4 result = glm::mat4(1.0f);
        result[0] = glm::vec4(x, 0.0f);
        result[1] = glm::vec4(y, 0.0f);
        result[2] = glm::vec4(z, 0.0f);
        result[3] = matrix[3];

        return result;
    }

    float CropBoxTool::wrapAngle(float angle) {
        while (angle < 0.0f)
            angle += 360.0f;
        while (angle >= 360.0f)
            angle -= 360.0f;
        return angle;
    }

} // namespace gs::visualizer
