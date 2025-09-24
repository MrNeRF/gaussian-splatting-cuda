/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/tools_panel.hpp"
#include "gui/panels/crop_box_panel.hpp"
#include "gui/panels/world_transform_panel.hpp"
#include <imgui.h>

namespace gs::gui::panels {

    void DrawToolsPanel(const UIContext& ctx) {

        // Draw crop box controls
        DrawCropBoxControls(ctx);

        // Draw world transform controls
        DrawWorldTransformControls(ctx);
    }

} // namespace gs::gui::panels
