#pragma once

#include "gui/ui_context.hpp"

// Forward declarations
namespace gs::visualizer {
    class ToolBase;
}

namespace gs::gui::panels {

    // Main function to draw the tools panel
    void DrawToolsPanel(const UIContext& ctx);

    // Helper functions for tool panel rendering
    namespace detail {
        // Draw a single tool's UI
        void DrawToolUI(const UIContext& ctx, gs::visualizer::ToolBase* tool);

        // Draw tool header with enable/disable checkbox
        bool DrawToolHeader(gs::visualizer::ToolBase* tool);

        // Get icon for tool (for future use)
        const char* GetToolIcon(const std::string_view& tool_name);
    } // namespace detail

} // namespace gs::gui::panels