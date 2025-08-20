#include "gui/panels/training_panel.hpp"
#include "core/events.hpp"
#include "gui/ui_widgets.hpp"
#include "visualizer_impl.hpp"
#include <imgui.h>

namespace gs::gui::panels {

    void DrawTrainingControls(const UIContext& ctx) {
        ImGui::Text("Training Control");
        ImGui::Separator();

        auto& state = TrainingPanelState::getInstance();

        // Direct call to TrainerManager - no state duplication
        auto* trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No trainer loaded");
            return;
        }

        // Get state directly from the single source of truth
        auto trainer_state = trainer_manager->getState();
        int current_iteration = trainer_manager->getCurrentIteration();
        float current_loss = trainer_manager->getCurrentLoss();

        // Render controls based on trainer state
        switch (trainer_state) {
        case TrainerManager::State::Idle:
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No trainer loaded");
            break;

        case TrainerManager::State::Ready:
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            if (ImGui::Button("Start Training", ImVec2(-1, 0))) {
                events::cmd::StartTraining{}.emit();
            }
            ImGui::PopStyleColor(2);
            break;

        case TrainerManager::State::Running:
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.5f, 0.1f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.6f, 0.2f, 1.0f));
            if (ImGui::Button("Pause", ImVec2(-1, 0))) {
                events::cmd::PauseTraining{}.emit();
            }
            ImGui::PopStyleColor(2);
            break;

        case TrainerManager::State::Paused:
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

        case TrainerManager::State::Completed:
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Training Complete!");
            break;

        case TrainerManager::State::Error:
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Training Error!");
            {
                auto error_msg = trainer_manager->getLastError();
                if (!error_msg.empty()) {
                    ImGui::TextWrapped("%s", error_msg.c_str());
                }
            }
            break;

        case TrainerManager::State::Stopping:
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Stopping...");
            break;
        }

        // Save checkpoint button (available during training)
        if (trainer_state == TrainerManager::State::Running ||
            trainer_state == TrainerManager::State::Paused) {

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

        // Helper to convert state to string
        const char* state_str = "Unknown";
        switch (trainer_state) {
        case TrainerManager::State::Idle: state_str = "Idle"; break;
        case TrainerManager::State::Ready: state_str = "Ready"; break;
        case TrainerManager::State::Running: state_str = "Running"; break;
        case TrainerManager::State::Paused: state_str = "Paused"; break;
        case TrainerManager::State::Stopping: state_str = "Stopping"; break;
        case TrainerManager::State::Completed: state_str = "Completed"; break;
        case TrainerManager::State::Error: state_str = "Error"; break;
        }

        ImGui::Text("Status: %s", state_str);
        ImGui::Text("Iteration: %d", current_iteration);
        ImGui::Text("Loss: %.6f", current_loss);
    }

} // namespace gs::gui::panels