/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"

namespace gs::gui::panels {

    // Main panel rendering
    void DrawMainPanel(const UIContext& ctx);

    // Individual sections
    void DrawWindowControls(const UIContext& ctx);
    void DrawRenderingSettings(const UIContext& ctx);
    void DrawProgressInfo(const UIContext& ctx);
    void DrawSystemConsoleButton(const UIContext& ctx);
} // namespace gs::gui::panels
