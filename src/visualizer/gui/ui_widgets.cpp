#include "gui/ui_widgets.hpp"
#include "core/events.hpp"
#include "visualizer_impl.hpp"
#include <format>
#include <imgui.h>

namespace gs::gui::widgets {

    bool SliderWithReset(const char* label, float* v, float min, float max, float reset_value) {
        bool changed = false;

        ImGui::PushItemWidth(200);
        std::string slider_label = std::format("##{}_slider", label);
        std::string display = std::format("{}={:.2f}", label, *v);
        changed |= ImGui::SliderFloat(slider_label.c_str(), v, min, max, display.c_str());
        ImGui::PopItemWidth();

        ImGui::SameLine();
        std::string reset_label = std::format("Reset##{}", label);
        if (ImGui::Button(reset_label.c_str(), ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            *v = reset_value;
            changed = true;
        }

        return changed;
    }

    bool DragFloat3WithReset(const char* label, float* v, float speed, float reset_value) {
        bool changed = false;

        ImGui::Text("%s:", label);
        ImGui::SameLine();

        std::string drag_label = std::format("##{}_drag", label);
        changed |= ImGui::DragFloat3(drag_label.c_str(), v, speed);

        ImGui::SameLine();
        std::string reset_label = std::format("Reset##{}", label);
        if (ImGui::Button(reset_label.c_str())) {
            v[0] = v[1] = v[2] = reset_value;
            changed = true;
        }

        return changed;
    }

    void HelpMarker(const char* desc) {
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    void TableRow(const char* label, const char* format, ...) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", label);
        ImGui::TableNextColumn();

        va_list args;
        va_start(args, format);
        ImGui::TextV(format, args);
        va_end(args);
    }

    void DrawProgressBar(float fraction, const char* overlay_text) {
        ImGui::ProgressBar(fraction, ImVec2(-1, 20), overlay_text);
    }

    void DrawLossPlot(const float* values, int count, float min_val, float max_val, const char* label) {
        ImGui::PlotLines("##Loss", values, count, 0, label, min_val, max_val, ImVec2(-1, 50));
    }

    void DrawModeStatus(const UIContext& ctx) {
        // Query scene info using the new event system
        events::query::SceneInfo response;
        bool has_response = false;

        // Set up a one-time handler for the response
        [[maybe_unused]] auto handler = events::query::SceneInfo::when([&response, &has_response](const auto& r) {
            response = r;
            has_response = true;
        });

        // Send the query
        events::query::GetSceneInfo{}.emit();

        if (!has_response || response.type == events::query::SceneInfo::Type::None) {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No data loaded");
            ImGui::Text("Use File Browser to load:");
            ImGui::BulletText("PLY file for viewing");
            ImGui::BulletText("Dataset for training");
            return;
        }

        switch (response.type) {
        case events::query::SceneInfo::Type::PLY:
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "PLY Viewer Mode");
            if (ctx.viewer->getCurrentPLYPath().has_filename()) {
                ImGui::Text("File: %s", ctx.viewer->getCurrentPLYPath().filename().string().c_str());
            }
            break;

        case events::query::SceneInfo::Type::Dataset:
            ImGui::TextColored(ImVec4(0.2f, 0.5f, 0.8f, 1.0f), "Training Mode");
            if (ctx.viewer->getCurrentDatasetPath().has_filename()) {
                ImGui::Text("Dataset: %s", ctx.viewer->getCurrentDatasetPath().filename().string().c_str());
            }
            break;

        default:
            break;
        }
    }

    const char* GetTrainerStateString(int state) {
        switch (state) {
        case 0: return "Idle";
        case 1: return "Ready";
        case 2: return "Running";
        case 3: return "Paused";
        case 4: return "Completed";
        case 5: return "Error";
        default: return "Unknown";
        }
    }

    std::string executeConsoleCommand([[maybe_unused]] const std::string& command,
                                      [[maybe_unused]] visualizer::VisualizerImpl* viewer) {
        // This is now handled in visualizer_impl.cpp
        return "Command execution moved to visualizer";
    }
} // namespace gs::gui::widgets