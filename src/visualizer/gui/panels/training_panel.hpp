#pragma once

#include "gui/ui_context.hpp"
#include <chrono>

namespace gs::gui::panels {

    void DrawTrainingControls(const UIContext& ctx);

    // Training panel state
    struct TrainingPanelState {
        bool save_in_progress = false;
        std::chrono::steady_clock::time_point save_start_time;

        static TrainingPanelState& getInstance() {
            static TrainingPanelState instance;
            return instance;
        }
    };
} // namespace gs::gui::panels
