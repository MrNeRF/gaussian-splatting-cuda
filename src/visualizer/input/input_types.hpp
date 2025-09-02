/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace gs {

    // Input event types shared across the system
    struct MouseButtonEvent {
        int button;
        int action;
        int mods;
        glm::dvec2 position;
    };

    struct MouseMoveEvent {
        glm::dvec2 position;
        glm::dvec2 delta;
    };

    struct MouseScrollEvent {
        double xoffset;
        double yoffset;
    };

    struct KeyEvent {
        int key;
        int scancode;
        int action;
        int mods;
    };

    struct FileDropEvent {
        std::vector<std::string> paths;
    };

    // Callback types
    using MouseButtonCallback = std::function<void(const MouseButtonEvent&)>;
    using MouseMoveCallback = std::function<void(const MouseMoveEvent&)>;
    using MouseScrollCallback = std::function<void(const MouseScrollEvent&)>;
    using KeyCallback = std::function<void(const KeyEvent&)>;
    using FileDropCallback = std::function<void(const FileDropEvent&)>;

} // namespace gs
