#include "gui/panels/world_transform_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>

namespace gs::gui::panels {

    void DrawWorldTransformControls(const UIContext& ctx) {
        auto render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager)
            return;

        if (!ImGui::CollapsingHeader("World Transform")) {
            return;
        }

        auto settings = render_manager->getSettings();
        bool settings_changed = false;

        // Store transform UI values statically
        static float rotation_degrees[3] = {0.0f, 0.0f, 0.0f};
        static float translation[3] = {0.0f, 0.0f, 0.0f};
        static bool transform_initialized = false;

        // Initialize from current transform if needed
        if (!transform_initialized) {
            glm::mat3 rot_mat = settings.world_transform.getRotationMat();
            glm::vec3 euler = glm::eulerAngles(glm::quat_cast(rot_mat));
            rotation_degrees[0] = glm::degrees(euler.x);
            rotation_degrees[1] = glm::degrees(euler.y);
            rotation_degrees[2] = glm::degrees(euler.z);

            glm::vec3 trans = settings.world_transform.getTranslation();
            translation[0] = trans.x;
            translation[1] = trans.y;
            translation[2] = trans.z;
            transform_initialized = true;
        }

        // Transform status
        if (!settings.world_transform.isIdentity()) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Transform Active");
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Identity Transform");
        }

        ImGui::Separator();

        // Rotation controls
        ImGui::Text("Rotation (degrees):");
        if (ImGui::DragFloat3("##world_rotation", rotation_degrees, 0.1f, -360.0f, 360.0f, "%.1f")) {
            settings_changed = true;
        }

        // Translation controls
        ImGui::Text("Translation:");
        if (ImGui::DragFloat3("##world_translation", translation, 0.01f, -100.0f, 100.0f, "%.3f")) {
            settings_changed = true;
        }

        if (settings_changed) {
            // Update transform from UI values
            glm::vec3 rot_rad(glm::radians(rotation_degrees[0]),
                              glm::radians(rotation_degrees[1]),
                              glm::radians(rotation_degrees[2]));
            settings.world_transform = geometry::EuclideanTransform(
                rot_rad.x, rot_rad.y, rot_rad.z,
                translation[0], translation[1], translation[2]);

            render_manager->updateSettings(settings);
        }

        ImGui::Separator();

        // Reset button
        if (ImGui::Button("Reset Transform", ImVec2(-1, 0))) {
            settings.world_transform = geometry::EuclideanTransform();
            rotation_degrees[0] = rotation_degrees[1] = rotation_degrees[2] = 0.0f;
            translation[0] = translation[1] = translation[2] = 0.0f;
            render_manager->updateSettings(settings);
        }

        // Show transform matrix info
        if (ImGui::TreeNode("Transform Matrix")) {
            glm::mat3 rot = settings.world_transform.getRotationMat();
            ImGui::Text("Rotation:");
            ImGui::Text("[%.3f, %.3f, %.3f]", rot[0][0], rot[1][0], rot[2][0]);
            ImGui::Text("[%.3f, %.3f, %.3f]", rot[0][1], rot[1][1], rot[2][1]);
            ImGui::Text("[%.3f, %.3f, %.3f]", rot[0][2], rot[1][2], rot[2][2]);

            glm::vec3 trans = settings.world_transform.getTranslation();
            ImGui::Separator();
            ImGui::Text("Translation: [%.3f, %.3f, %.3f]", trans.x, trans.y, trans.z);

            ImGui::TreePop();
        }
    }

} // namespace gs::gui::panels
