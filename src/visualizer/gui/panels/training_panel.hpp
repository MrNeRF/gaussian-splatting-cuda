/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "gui/ui_context.hpp"
#include "gui/windows/save_project_browser.hpp"
#include <chrono>
#include <unordered_map>

namespace gs::gui::panels {

    void DrawTrainingControls(const UIContext& ctx);
    void DrawTrainingParameters(const UIContext& ctx);

    // Training panel state
    struct TrainingPanelState {
        bool save_in_progress = false;
        std::chrono::steady_clock::time_point save_start_time;

        // Save project browser
        bool show_save_browser = false;
        SaveProjectBrowser save_browser;

        // Store parameters for each strategy separately
        std::unordered_map<std::string, param::OptimizationParameters> strategy_params_cache;

        // Track the last active strategy
        std::string last_active_strategy;

        static TrainingPanelState& getInstance() {
            static TrainingPanelState instance;
            return instance;
        }
    };
} // namespace gs::gui::panels