#pragma once
#include <cstdint>

namespace gs {

    // Input priority levels - higher values = higher priority
    enum class InputPriority : int32_t {
        Blocked = 10000, // Input is blocked (e.g., during loading)
        System = 1000,   // System-level shortcuts (e.g., ESC to quit)
        Modal = 900,     // Modal dialogs that must be addressed
        GUI = 800,       // Regular GUI elements (ImGui windows, buttons, etc.)
        Tools = 700,     // Interactive tools (crop box, etc.)
        Camera = 600,    // Camera controls
        Scene = 500,     // Scene interaction
        Default = 0      // Default/fallback priority
    };

    // Helper to check if one priority should handle before another
    constexpr bool has_higher_priority(InputPriority a, InputPriority b) {
        return static_cast<int32_t>(a) > static_cast<int32_t>(b);
    }

} // namespace gs
