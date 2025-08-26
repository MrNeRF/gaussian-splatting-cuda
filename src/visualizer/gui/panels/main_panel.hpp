#pragma once

#include "gui/ui_context.hpp"

namespace gs::gui::panels {

    // Main panel rendering
    void DrawMainPanel(const UIContext& ctx);

    // Individual sections
    void DrawWindowControls(const UIContext& ctx);
    void DrawRenderingSettings(const UIContext& ctx);
    void DrawProgressInfo(const UIContext& ctx);
} // namespace gs::gui::panels
