#include "gui/ui_widgets.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"
#include "visualizer_impl.hpp"
#include <cstdarg>
#include <imgui.h>

namespace gs::gui::widgets {

    bool SliderWithReset(const char* label, float* v, float min, float max, float reset_value) {
        bool changed = ImGui::SliderFloat(label, v, min, max);

        ImGui::SameLine();
        ImGui::PushID(label);
        if (ImGui::Button("Reset")) {
            *v = reset_value;
            changed = true;
        }
        ImGui::PopID();

        return changed;
    }

    bool DragFloat3WithReset(const char* label, float* v, float speed, float reset_value) {
        bool changed = ImGui::DragFloat3(label, v, speed);

        ImGui::SameLine();
        ImGui::PushID(label);
        if (ImGui::Button("Reset")) {
            v[0] = v[1] = v[2] = reset_value;
            changed = true;
        }
        ImGui::PopID();

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
        ImGui::ProgressBar(fraction, ImVec2(-1, 0), overlay_text);
    }

    void DrawLossPlot(const float* values, int count, float min_val, float max_val, const char* label) {
        if (count <= 0)
            return;

        // Simple line plot using ImGui
        ImGui::PlotLines(
            label,
            values,
            count,
            0,
            nullptr,
            min_val,
            max_val,
            ImVec2(0, 80));
    }

    void DrawModeStatus(const UIContext& ctx) {
        auto* scene_manager = ctx.viewer->getSceneManager();
        if (!scene_manager) {
            ImGui::Text("Mode: Unknown");
            return;
        }

        const char* mode_str = "Unknown";
        ImVec4 mode_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);

        SceneManager::ViewerMode mode = scene_manager->getCurrentMode(); // Use SceneManager:: namespace
        switch (mode) {
        case SceneManager::ViewerMode::Empty:
            mode_str = "Empty";
            mode_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
            break;
        case SceneManager::ViewerMode::PLYViewer:
            mode_str = "PLY Viewer";
            mode_color = ImVec4(0.2f, 0.6f, 1.0f, 1.0f);
            break;
        case SceneManager::ViewerMode::Training: {
            auto state = scene_manager->getState();
            if (auto* training = std::get_if<SceneManager::TrainingState>(&state)) {
                if (training->is_running) {
                    mode_str = "Training";
                    mode_color = ImVec4(1.0f, 0.6f, 0.2f, 1.0f);
                } else {
                    mode_str = "Dataset (Ready)";
                    mode_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
                }
            }
        } break;
        }

        ImGui::TextColored(mode_color, "Mode: %s", mode_str);

        auto info = scene_manager->getSceneInfo();
        if (info.num_gaussians > 0) {
            ImGui::Text("Gaussians: %zu", info.num_gaussians);
        }

        if (info.source_type == "PLY" && info.num_nodes > 0) {
            ImGui::Text("PLY Models: %zu", info.num_nodes);
        }

        if (scene_manager->isTraining()) {
            auto state = scene_manager->getState();
            if (auto* training = std::get_if<SceneManager::TrainingState>(&state)) {
                if (training->current_iteration > 0) {
                    ImGui::Text("Iteration: %d", training->current_iteration);
                }
            }
        }
    }
} // namespace gs::gui::widgets