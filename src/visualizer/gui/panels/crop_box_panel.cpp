#include "gui/panels/crop_box_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "rendering/render_bounding_box.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

namespace gs::gui::panels {

    // Apply rotation to crop box
    static void updateRotationMatrix(RenderBoundingBox* crop_box,
                                     float delta_rot_x, float delta_rot_y, float delta_rot_z) {
        if (!crop_box)
            return;

        float rad_x = glm::radians(delta_rot_x);
        float rad_y = glm::radians(delta_rot_y);
        float rad_z = glm::radians(delta_rot_z);

        geometry::EuclideanTransform rotate(rad_x, rad_y, rad_z, 0.0f, 0.0f, 0.0f);

        glm::vec3 center = crop_box->getLocalCenter();

        geometry::EuclideanTransform translate_to_origin(-center);
        geometry::EuclideanTransform translate_back = translate_to_origin.inv();

        geometry::EuclideanTransform current_transform = crop_box->getworld2BBox();
        auto rotation_transform = translate_back * rotate * translate_to_origin * current_transform;

        crop_box->setworld2BBox(rotation_transform);
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
                crop_box->setworld2BBox(geometry::EuclideanTransform());
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
