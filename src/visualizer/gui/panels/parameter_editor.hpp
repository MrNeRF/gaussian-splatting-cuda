/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include <imgui.h>
#include <string>

namespace gs::gui::panels {

    class ParameterEditor {
    public:
        ParameterEditor(param::OptimizationParameters& params, bool can_edit);
        ~ParameterEditor();

        // Start a new table section
        ParameterEditor& BeginSection(const char* table_name);
        void EndSection();

        // Add a table row and prepare for input
        ParameterEditor& Row(const char* label);

        // Parameter editing methods
        ParameterEditor& Int(const char* id, const char* key, int default_val,
                             int step1 = 100, int step2 = 1000, int min = 0, int max = 1000000);
        ParameterEditor& Float(const char* id, const char* key, float default_val,
                               float step1 = 0.001f, float step2 = 0.01f,
                               const char* format = "%.4f", float min = -FLT_MAX, float max = FLT_MAX);
        ParameterEditor& Bool(const char* id, const char* key, bool default_val);
        ParameterEditor& Slider(const char* id, const char* key, float default_val,
                                float min, float max, const char* format = "%.2f");
        ParameterEditor& SliderInt(const char* id, const char* key, int default_val,
                                   int min, int max);
        ParameterEditor& Combo(const char* id, const char* key, const char** options,
                               int count, const std::string& default_val);

        // Check if any parameter changed
        bool changed() const { return changed_; }

    private:
        void ApplyOverride(const std::string& key, const nlohmann::json& value);

        param::OptimizationParameters& params_;
        bool can_edit_;
        bool changed_ = false;
        bool in_table_ = false;
        bool row_started_ = false;
    };

} // namespace gs::gui::panels
