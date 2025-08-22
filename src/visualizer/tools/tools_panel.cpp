#include "gui/panels/tools_panel.hpp"
#include "core/events.hpp"
#include "gui/panels/crop_box_panel.hpp"
#include "gui/panels/world_transform_panel.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer_impl.hpp"
#include <imgui.h>

namespace gs::gui::panels {

    void DrawToolsPanel(const UIContext& ctx) {
        ImGui::Text("Visualization Tools");
        ImGui::Separator();

        // Translation Gizmo
        if (ImGui::CollapsingHeader("Translation Gizmo")) {
            auto* render_manager = ctx.viewer->getRenderingManager();
            if (render_manager) {
                auto settings = render_manager->getSettings();

                // Toggle for the gizmo
                bool show_gizmo = settings.show_translation_gizmo;
                if (ImGui::Checkbox("Enable Translation Gizmo", &show_gizmo)) {
                    settings.show_translation_gizmo = show_gizmo;
                    render_manager->updateSettings(settings);

                    // Emit appropriate event
                    if (show_gizmo) {
                        events::tools::ToolEnabled{.tool_name = "Translation Gizmo"}.emit();
                    } else {
                        events::tools::ToolDisabled{.tool_name = "Translation Gizmo"}.emit();
                    }
                }

                if (show_gizmo) {
                    // Gizmo scale
                    if (ImGui::SliderFloat("Gizmo Scale", &settings.gizmo_scale, 0.5f, 3.0f)) {
                        render_manager->updateSettings(settings);
                    }

                    ImGui::Separator();
                    ImGui::Text("Controls:");
                    ImGui::BulletText("Drag arrows: Move along axis");
                    ImGui::BulletText("Drag planes: Move in 2D plane");
                    ImGui::BulletText("  Blue plane: XY movement");
                    ImGui::BulletText("  Green plane: XZ movement");
                    ImGui::BulletText("  Red plane: YZ movement");
                    ImGui::BulletText("R: Reset position");

                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                                       "Note: Gizmo modifies world transform\n"
                                       "Changes are reflected in World Transform panel");
                }
            }
        }

        ImGui::Separator();

        // Draw crop box controls
        DrawCropBoxControls(ctx);

        // Draw world transform controls
        DrawWorldTransformControls(ctx);
    }

} // namespace gs::gui::panels
