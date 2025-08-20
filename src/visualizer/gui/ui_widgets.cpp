#include "gui/ui_widgets.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"
#include "visualizer_impl.hpp"
#include <cstdarg>
#include <format>
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
        // FIX: Direct call instead of query event
        auto* scene_manager = ctx.viewer->getSceneManager();
        if (!scene_manager) {
            ImGui::Text("Mode: Unknown");
            return;
        }

        auto state = scene_manager->getCurrentState();

        const char* mode_str = "Unknown";
        ImVec4 mode_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);

        switch (state.type) {
        case SceneManager::SceneType::None:
            mode_str = "Empty";
            mode_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
            break;
        case SceneManager::SceneType::PLY:
            mode_str = "PLY Viewer";
            mode_color = ImVec4(0.2f, 0.6f, 1.0f, 1.0f);
            break;
        case SceneManager::SceneType::Dataset:
            if (state.is_training) {
                mode_str = "Training";
                mode_color = ImVec4(1.0f, 0.6f, 0.2f, 1.0f);
            } else {
                mode_str = "Dataset (Ready)";
                mode_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
            }
            break;
        }

        ImGui::TextColored(mode_color, "Mode: %s", mode_str);

        if (state.num_gaussians > 0) {
            ImGui::Text("Gaussians: %zu", state.num_gaussians);
        }

        if (state.type == SceneManager::SceneType::PLY && state.num_plys > 0) {
            ImGui::Text("PLY Models: %zu", state.num_plys);
        }

        if (state.training_iteration.has_value()) {
            ImGui::Text("Iteration: %d", *state.training_iteration);
        }
    }

    const char* GetTrainerStateString(int state) {
        // Helper function for trainer state to string conversion
        switch (static_cast<TrainerManager::State>(state)) {
        case TrainerManager::State::Idle: return "Idle";
        case TrainerManager::State::Ready: return "Ready";
        case TrainerManager::State::Running: return "Running";
        case TrainerManager::State::Paused: return "Paused";
        case TrainerManager::State::Stopping: return "Stopping";
        case TrainerManager::State::Completed: return "Completed";
        case TrainerManager::State::Error: return "Error";
        default: return "Unknown";
        }
    }

    std::string executeConsoleCommand([[maybe_unused]] const std::string& command,
                                      [[maybe_unused]] visualizer::VisualizerImpl* viewer) {
        // This is now handled in visualizer_impl.cpp
        return "Command execution moved to visualizer";
    }
} // namespace gs::gui::widgets