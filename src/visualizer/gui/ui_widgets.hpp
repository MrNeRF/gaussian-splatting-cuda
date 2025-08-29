/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include <glm/glm.hpp>
#include <string>

namespace gs::gui::widgets {

    // Reusable UI widgets
    bool SliderWithReset(const char* label, float* v, float min, float max, float reset_value);
    bool DragFloat3WithReset(const char* label, float* v, float speed, float reset_value);
    void HelpMarker(const char* desc);
    void TableRow(const char* label, const char* format, ...);

    // Progress display
    void DrawProgressBar(float fraction, const char* overlay_text);
    void DrawLossPlot(const float* values, int count, float min_val, float max_val, const char* label);
    void DrawModeStatusWithContentSwitch(const UIContext& ctx);
    // Mode display helpers
    void DrawModeStatus(const UIContext& ctx);
} // namespace gs::gui::widgets