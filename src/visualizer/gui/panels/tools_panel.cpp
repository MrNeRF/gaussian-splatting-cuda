#include "gui/panels/tools_panel.hpp"
#include "gui/panels/crop_box_panel.hpp"
#include <imgui.h>

namespace gs::gui::panels {

    void DrawToolsPanel(const UIContext& ctx) {
        ImGui::Text("Visualization");
        ImGui::Separator();

        // Draw crop box controls
        DrawCropBoxControls(ctx);
    }

} // namespace gs::gui::panels