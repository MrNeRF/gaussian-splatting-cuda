/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"

namespace gs::gui::panels {

    void DrawCropBoxControls(const UIContext& ctx);

    // Crop box state
    struct CropBoxState {
        bool show_crop_box = false;
        bool use_crop_box = false;

        static CropBoxState& getInstance() {
            static CropBoxState instance;
            return instance;
        }
    };

    // Access for GuiManager - Add this declaration
    const CropBoxState& getCropBoxState();
} // namespace gs::gui::panels