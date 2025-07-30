#include "gui/panels/crop_box_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "rendering/render_bounding_box.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

namespace gs::gui::panels {

    // Helper function to wrap angles
    static float wrapAngle(float angle) {
        while (angle < 0.0f)
            angle += 360.0f;
        while (angle >= 360.0f)
            angle -= 360.0f;
        return angle;
    }

    // Orthonormalize rotation matrix
    static glm::mat4 OrthonormalizeRotation(const glm::mat4& matrix) {
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

    // Apply rotation to crop box
    static void updateRotationMatrix(RenderBoundingBox* crop_box,
                                     float delta_rot_x, float delta_rot_y, float delta_rot_z) {
        if (!crop_box)
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

        glm::vec3 center = crop_box->getLocalCenter();

        glm::mat4 translate_to_origin = glm::mat4(1.0f);
        translate_to_origin[3][0] = -center.x;
        translate_to_origin[3][1] = -center.y;
        translate_to_origin[3][2] = -center.z;

        glm::mat4 translate_back = glm::mat4(1.0f);
        translate_back[3][0] = center.x;
        translate_back[3][1] = center.y;
        translate_back[3][2] = center.z;

        glm::mat4 rotation_transform = translate_back * combined_rotation * translate_to_origin;
        glm::mat4 curr_world2bbox = crop_box->getworld2BBox();
        glm::mat4 final_transform = rotation_transform * curr_world2bbox;
        final_transform = OrthonormalizeRotation(final_transform);

        crop_box->setworld2BBox(final_transform);
    }

    void DrawCropBoxControls(const UIContext& ctx) {
        auto& state = CropBoxState::getInstance();

        if (!ImGui::CollapsingHeader("Crop Box")) {
            return;
        }

        auto crop_box = ctx.viewer->getCropBox();
        if (!crop_box)
            return;

        ImGui::Checkbox("Show Crop Box", &state.show_crop_box);
        ImGui::Checkbox("Use Crop Box", &state.use_crop_box);

        if (state.show_crop_box && crop_box->isInitialized()) {
            // Appearance controls
            static float bbox_color[3] = {1.0f, 1.0f, 0.0f};
            if (ImGui::ColorEdit3("Box Color", bbox_color)) {
                crop_box->setColor(glm::vec3(bbox_color[0], bbox_color[1], bbox_color[2]));
            }

            static float line_width = 2.0f;
            float available_width = ImGui::GetContentRegionAvail().x;
            float button_width = 120.0f;
            float slider_width = available_width - button_width - ImGui::GetStyle().ItemSpacing.x;

            ImGui::SetNextItemWidth(slider_width);
            if (ImGui::SliderFloat("Line Width", &line_width, 0.5f, 10.0f)) {
                crop_box->setLineWidth(line_width);
            }

            if (ImGui::Button("Reset to Default")) {
                crop_box->setBounds(glm::vec3(-1.0f), glm::vec3(1.0f));
                crop_box->setworld2BBox(glm::mat4(1.0));
            }

            // Rotation controls
            if (ImGui::TreeNode("Rotation")) {
                ImGui::Text("Ctrl+click for faster steps");
                ImGui::Text("Rotation around crop box axes:");

                const float rotation_step = 1.0f;
                const float rotation_step_fast = 15.0f;

                static float rotate_timer_x = 0.0f;
                static float rotate_timer_y = 0.0f;
                static float rotate_timer_z = 0.0f;

                float step = ImGui::GetIO().KeyCtrl ? rotation_step_fast : rotation_step;
                float repeat_rate = 0.05f;

                float diff_x = 0, diff_y = 0, diff_z = 0;

                // X-axis rotation
                ImGui::Text("X-axis:");
                ImGui::SameLine();
                ImGui::Text("RotX");

                if (ImGui::ArrowButton("##RotX_Up", ImGuiDir_Up)) {
                    diff_x = step;
                    rotate_timer_x = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_x += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_x >= repeat_rate) {
                        diff_x = step;
                        rotate_timer_x = 0.0f;
                    }
                }

                ImGui::SameLine();
                if (ImGui::ArrowButton("##RotX_Down", ImGuiDir_Down)) {
                    diff_x = -step;
                    rotate_timer_x = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_x += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_x >= repeat_rate) {
                        diff_x = -step;
                        rotate_timer_x = 0.0f;
                    }
                }

                // Y-axis rotation
                ImGui::Text("Y-axis:");
                ImGui::SameLine();
                ImGui::Text("RotY");

                if (ImGui::ArrowButton("##RotY_Up", ImGuiDir_Up)) {
                    diff_y = step;
                    rotate_timer_y = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_y += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_y >= repeat_rate) {
                        diff_y = step;
                        rotate_timer_y = 0.0f;
                    }
                }

                ImGui::SameLine();
                if (ImGui::ArrowButton("##RotY_Down", ImGuiDir_Down)) {
                    diff_y = -step;
                    rotate_timer_y = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_y += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_y >= repeat_rate) {
                        diff_y = -step;
                        rotate_timer_y = 0.0f;
                    }
                }

                // Z-axis rotation
                ImGui::Text("Z-axis:");
                ImGui::SameLine();
                ImGui::Text("RotZ");

                if (ImGui::ArrowButton("##RotZ_Up", ImGuiDir_Up)) {
                    diff_z = step;
                    rotate_timer_z = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_z += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_z >= repeat_rate) {
                        diff_z = step;
                        rotate_timer_z = 0.0f;
                    }
                }

                ImGui::SameLine();
                if (ImGui::ArrowButton("##RotZ_Down", ImGuiDir_Down)) {
                    diff_z = -step;
                    rotate_timer_z = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_z += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_z >= repeat_rate) {
                        diff_z = -step;
                        rotate_timer_z = 0.0f;
                    }
                }

                if (diff_x != 0 || diff_y != 0 || diff_z != 0) {
                    updateRotationMatrix(crop_box.get(), diff_x, diff_y, diff_z);
                }

                ImGui::TreePop();
            }

            // Bounds controls
            if (ImGui::TreeNode("Bounds")) {
                glm::vec3 current_min = crop_box->getMinBounds();
                glm::vec3 current_max = crop_box->getMaxBounds();

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
                    crop_box->setBounds(
                        glm::vec3(min_bounds[0], min_bounds[1], min_bounds[2]),
                        glm::vec3(max_bounds[0], max_bounds[1], max_bounds[2]));
                }

                // Display info
                glm::vec3 center = crop_box->getCenter();
                glm::vec3 size = crop_box->getSize();

                ImGui::Text("Center: (%.3f, %.3f, %.3f)", center.x, center.y, center.z);
                ImGui::Text("Size: (%.3f, %.3f, %.3f)", size.x, size.y, size.z);

                ImGui::TreePop();
            }
        }
    }

    const CropBoxState& getCropBoxState() {
        return CropBoxState::getInstance();
    }
} // namespace gs::gui::panels
