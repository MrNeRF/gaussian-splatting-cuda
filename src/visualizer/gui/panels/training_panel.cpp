#include "gui/panels/training_panel.hpp"
#include "core/events.hpp"
#include "gui/ui_widgets.hpp"
#include <imgui.h>

namespace gs::gui::panels {

    void DrawTrainingControls([[maybe_unused]] const UIContext& ctx) {
        ImGui::Text("Training Control");
        ImGui::Separator();

        auto& state = TrainingPanelState::getInstance();

        // Query trainer state using the new event system
        events::query::TrainerState response;
        bool has_response = false;

        // Set up a one-time handler for the response
        [[maybe_unused]] auto handler = events::query::TrainerState::when([&response, &has_response](const auto& r) {
            response = r;
            has_response = true;
        });

        // Send the query
        events::query::GetTrainerState{}.emit();

        // Wait a bit for response (in a real app, this would be async)
        // For now, we'll assume the response is immediate

        if (!has_response) {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No trainer loaded");
            return;
        }

        // Render controls based on state
        switch (response.state) {
        case events::query::TrainerState::State::Idle:
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No trainer loaded");
            break;

        case events::query::TrainerState::State::Ready:
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            if (ImGui::Button("Start Training", ImVec2(-1, 0))) {
                events::cmd::StartTraining{}.emit();
            }
            ImGui::PopStyleColor(2);
            break;

        case events::query::TrainerState::State::Running:
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.5f, 0.1f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.6f, 0.2f, 1.0f));
            if (ImGui::Button("Pause", ImVec2(-1, 0))) {
                events::cmd::PauseTraining{}.emit();
            }
            ImGui::PopStyleColor(2);
            break;

        case events::query::TrainerState::State::Paused:
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            if (ImGui::Button("Resume", ImVec2(-1, 0))) {
                events::cmd::ResumeTraining{}.emit();
            }
            ImGui::PopStyleColor(2);

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button("Stop Permanently", ImVec2(-1, 0))) {
                events::cmd::StopTraining{}.emit();
            }
            ImGui::PopStyleColor(2);
            break;

        case events::query::TrainerState::State::Completed:
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Training Complete!");
            break;

        case events::query::TrainerState::State::Error:
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Training Error!");
            if (response.error_message) {
                ImGui::TextWrapped("%s", response.error_message->c_str());
            }
            break;
        }

        // Save checkpoint button (available during training)
        if (response.state == events::query::TrainerState::State::Running ||
            response.state == events::query::TrainerState::State::Paused) {

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.4f, 0.7f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
            if (ImGui::Button("Save Checkpoint", ImVec2(-1, 0))) {
                events::cmd::SaveCheckpoint{}.emit();
                state.save_in_progress = true;
                state.save_start_time = std::chrono::steady_clock::now();
            }
            ImGui::PopStyleColor(2);
        }

        // Save feedback
        if (state.save_in_progress) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - state.save_start_time)
                               .count();
            if (elapsed < 2000) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Checkpoint saved!");
            } else {
                state.save_in_progress = false;
            }
        }

        // Status display
        ImGui::Separator();
        ImGui::Text("Status: %s", widgets::GetTrainerStateString(static_cast<int>(response.state)));
        ImGui::Text("Iteration: %d", response.current_iteration);
        ImGui::Text("Loss: %.6f", response.current_loss);
    }
} // namespace gs::gui::panels