#include "gui/panels/training_panel.hpp"
#include "core/event_response_handler.hpp"
#include "core/events.hpp"
#include "gui/ui_widgets.hpp"
#include <format>
#include <imgui.h>

namespace gs::gui::panels {

    void DrawTrainingControls(const UIContext& ctx) {
        ImGui::Text("Training Control");
        ImGui::Separator();

        auto& state = TrainingPanelState::getInstance();

        // Query trainer state
        EventResponseHandler<QueryTrainerStateRequest, QueryTrainerStateResponse> handler(ctx.event_bus);
        auto response = handler.querySync(QueryTrainerStateRequest{});

        if (!response) {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No trainer loaded");
            return;
        }

        // Render controls based on state
        switch (response->state) {
        case QueryTrainerStateResponse::State::Idle:
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No trainer loaded");
            break;
        case QueryTrainerStateResponse::State::Ready:
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            if (ImGui::Button("Start Training", ImVec2(-1, 0))) {
                ctx.event_bus->publish(StartTrainingCommand{});
            }
            ImGui::PopStyleColor(2);
            break;

        case QueryTrainerStateResponse::State::Running:
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.5f, 0.1f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.6f, 0.2f, 1.0f));
            if (ImGui::Button("Pause", ImVec2(-1, 0))) {
                ctx.event_bus->publish(PauseTrainingCommand{});
            }
            ImGui::PopStyleColor(2);
            break;

        case QueryTrainerStateResponse::State::Paused:
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            if (ImGui::Button("Resume", ImVec2(-1, 0))) {
                ctx.event_bus->publish(ResumeTrainingCommand{});
            }
            ImGui::PopStyleColor(2);

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button("Stop Permanently", ImVec2(-1, 0))) {
                ctx.event_bus->publish(StopTrainingCommand{});
            }
            ImGui::PopStyleColor(2);
            break;

        case QueryTrainerStateResponse::State::Stopping:
            ImGui::TextColored(ImVec4(0.7f, 0.5f, 0.1f, 1.0f), "Stopping training...");
            break;

        case QueryTrainerStateResponse::State::Completed:
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Training Complete!");
            break;

        case QueryTrainerStateResponse::State::Error:
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Training Error!");
            if (response->error_message) {
                ImGui::TextWrapped("%s", response->error_message->c_str());
            }
            break;
        }

        // Save checkpoint button (available during training)
        if (response->state == QueryTrainerStateResponse::State::Running ||
            response->state == QueryTrainerStateResponse::State::Paused) {

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.4f, 0.7f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
            if (ImGui::Button("Save Checkpoint", ImVec2(-1, 0))) {
                ctx.event_bus->publish(SaveCheckpointCommand{});
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
        ImGui::Text("Status: %s", widgets::GetTrainerStateString(static_cast<int>(response->state)));
        ImGui::Text("Iteration: %d", response->current_iteration);
        ImGui::Text("Loss: %.6f", response->current_loss);
    }
} // namespace gs::gui::panels
