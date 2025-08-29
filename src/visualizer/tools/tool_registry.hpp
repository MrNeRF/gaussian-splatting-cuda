/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tool_base.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gs::visualizer {

    class ToolRegistry {
    public:
        using ToolFactory = std::function<std::unique_ptr<ToolBase>()>;

        // Register a tool type
        template <Tool T>
        void registerTool() {
            auto factory = []() -> std::unique_ptr<ToolBase> {
                return std::make_unique<T>();
            };

            auto temp = factory();
            std::string name(temp->getName());
            tool_factories_[name] = std::move(factory);
        }

        // Create tool instance
        std::unique_ptr<ToolBase> createTool(const std::string& name) {
            auto it = tool_factories_.find(name);
            if (it != tool_factories_.end()) {
                return it->second();
            }
            return nullptr;
        }

        // Get available tools
        std::vector<std::string> getAvailableTools() const {
            std::vector<std::string> names;
            for (const auto& [name, _] : tool_factories_) {
                names.push_back(name);
            }
            return names;
        }

        // Check if tool is registered
        bool hasToolFactory(const std::string& name) const {
            return tool_factories_.find(name) != tool_factories_.end();
        }

    private:
        std::unordered_map<std::string, ToolFactory> tool_factories_;
    };

} // namespace gs::visualizer
