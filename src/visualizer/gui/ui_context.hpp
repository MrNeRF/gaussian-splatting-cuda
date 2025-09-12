/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace gs {
    // Forward declarations
    namespace visualizer {
        class VisualizerImpl;
    }

    namespace gui {
        class ScriptingConsole;
        class FileBrowser;
        class DialogBox;

        // Shared context passed to all UI functions
        struct UIContext {
            visualizer::VisualizerImpl* viewer;
            ScriptingConsole* console;
            FileBrowser* file_browser;
            DialogBox* dialog_box;
            std::unordered_map<std::string, bool>* window_states;
        };
    } // namespace gui
} // namespace gs