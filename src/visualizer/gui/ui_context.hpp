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

        // Shared context passed to all UI functions
        struct UIContext {
            visualizer::VisualizerImpl* viewer;
            ScriptingConsole* console;
            FileBrowser* file_browser;
            std::unordered_map<std::string, bool>* window_states;
        };
    } // namespace gui
} // namespace gs