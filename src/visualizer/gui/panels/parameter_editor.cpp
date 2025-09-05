/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/parameter_editor.hpp"

namespace gs::gui::panels {

    ParameterEditor::ParameterEditor(param::OptimizationParameters& params, bool can_edit)
        : params_(params),
          can_edit_(can_edit) {
    }

    ParameterEditor::~ParameterEditor() {
        if (in_table_) {
            EndSection();
        }
    }

    ParameterEditor& ParameterEditor::BeginSection(const char* table_name) {
        if (ImGui::BeginTable(table_name, 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
            in_table_ = true;
        }
        return *this;
    }

    void ParameterEditor::EndSection() {
        if (in_table_) {
            ImGui::EndTable();
            in_table_ = false;
        }
    }

    ParameterEditor& ParameterEditor::Row(const char* label) {
        if (in_table_) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", label);
            ImGui::TableNextColumn();
            row_started_ = true;
        }
        return *this;
    }

    void ParameterEditor::ApplyOverride(const std::string& key, const nlohmann::json& value) {
        nlohmann::json overrides;
        overrides[key] = value;
        params_.params = params_.params.with_overrides(overrides);
        changed_ = true;
    }

    ParameterEditor& ParameterEditor::Int(const char* id, const char* key, int default_val,
                                          int step1, int step2, int min, int max) {
        if (can_edit_) {
            ImGui::PushItemWidth(-1);
            int value = params_.params.get<int>(key, default_val);
            if (ImGui::InputInt(id, &value, step1, step2)) {
                if (value >= min && value <= max) {
                    ApplyOverride(key, value);
                }
            }
            ImGui::PopItemWidth();
        } else {
            ImGui::Text("%d", params_.params.get<int>(key, default_val));
        }
        return *this;
    }

    ParameterEditor& ParameterEditor::Float(const char* id, const char* key, float default_val,
                                            float step1, float step2, const char* format,
                                            float min, float max) {
        if (can_edit_) {
            ImGui::PushItemWidth(-1);
            float value = params_.params.get<float>(key, default_val);
            if (ImGui::InputFloat(id, &value, step1, step2, format)) {
                if (value >= min && value <= max) {
                    ApplyOverride(key, value);
                }
            }
            ImGui::PopItemWidth();
        } else {
            ImGui::Text(format, params_.params.get<float>(key, default_val));
        }
        return *this;
    }

    ParameterEditor& ParameterEditor::Bool(const char* id, const char* key, bool default_val) {
        bool value = params_.params.get<bool>(key, default_val);
        if (can_edit_) {
            if (ImGui::Checkbox(id, &value)) {
                ApplyOverride(key, value);
            }
        } else {
            ImGui::Text("%s", value ? "Yes" : "No");
        }
        return *this;
    }

    ParameterEditor& ParameterEditor::Slider(const char* id, const char* key, float default_val,
                                             float min, float max, const char* format) {
        if (can_edit_) {
            ImGui::PushItemWidth(-1);
            float value = params_.params.get<float>(key, default_val);
            if (ImGui::SliderFloat(id, &value, min, max, format)) {
                ApplyOverride(key, value);
            }
            ImGui::PopItemWidth();
        } else {
            ImGui::Text(format, params_.params.get<float>(key, default_val));
        }
        return *this;
    }

    ParameterEditor& ParameterEditor::SliderInt(const char* id, const char* key, int default_val,
                                                int min, int max) {
        if (can_edit_) {
            ImGui::PushItemWidth(-1);
            int value = params_.params.get<int>(key, default_val);
            if (ImGui::SliderInt(id, &value, min, max)) {
                ApplyOverride(key, value);
            }
            ImGui::PopItemWidth();
        } else {
            ImGui::Text("%d", params_.params.get<int>(key, default_val));
        }
        return *this;
    }

    ParameterEditor& ParameterEditor::Combo(const char* id, const char* key, const char** options,
                                            int count, const std::string& default_val) {
        if (can_edit_) {
            ImGui::PushItemWidth(-1);
            std::string current_val = params_.params.get<std::string>(key, default_val);
            int current_idx = 0;
            for (int i = 0; i < count; ++i) {
                if (current_val == options[i]) {
                    current_idx = i;
                    break;
                }
            }
            if (ImGui::Combo(id, &current_idx, options, count)) {
                ApplyOverride(key, std::string(options[current_idx]));
            }
            ImGui::PopItemWidth();
        } else {
            ImGui::Text("%s", params_.params.get<std::string>(key, default_val).c_str());
        }
        return *this;
    }

} // namespace gs::gui::panels
