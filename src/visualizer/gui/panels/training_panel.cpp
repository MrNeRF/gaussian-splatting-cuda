/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/training_panel.hpp"
#include "core/events.hpp"
#include "gui/ui_widgets.hpp"
#include "visualizer_impl.hpp"
#include <imgui.h>

namespace gs::gui::panels {

    void SaveProjectButton(const UIContext& ctx, TrainingPanelState& state) {
        // Add Save Project button
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.3f, 0.9f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.4f, 1.0f, 1.0f));
        if (ImGui::Button("Save Project", ImVec2(-1, 0))) {
            auto project = ctx.viewer->getProject();
            if (project) {
                if (project->getIsTempProject()) {
                    state.show_save_browser = true;
                } else {
                    events::cmd::SaveProject{project->getProjectOutputFolder().string()}.emit();
                }
            }
        }
        ImGui::PopStyleColor(2);
    }

    void DrawTrainingParameters(const UIContext& ctx) {
        auto* trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            return;
        }

        const auto* trainer = trainer_manager->getTrainer();
        if (!trainer) {
            return;
        }

        const auto& params = trainer->getParams();

        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 12.0f);

        // Dataset Parameters
        if (ImGui::TreeNode("Dataset")) {
            if (ImGui::BeginTable("DatasetTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Path:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", params.dataset.data_path.filename().string().c_str());

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Images:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", params.dataset.images.c_str());

                if (params.dataset.resize_factor != -1) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Resize Factor:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", params.dataset.resize_factor);
                }

                // Only show test_every if evaluation is enabled
                if (params.optimization.enable_eval) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Test Every:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", params.dataset.test_every);
                }

                if (!params.dataset.output_path.empty()) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Output:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", params.dataset.output_path.filename().string().c_str());
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Optimization Parameters
        if (ImGui::TreeNode("Optimization")) {
            if (ImGui::BeginTable("OptimizationTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Iterations:");
                ImGui::TableNextColumn();
                ImGui::Text("%zu", params.optimization.iterations);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Strategy:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", params.optimization.strategy.c_str());

                // Learning Rates section
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Learning Rates:");
                ImGui::TableNextColumn();

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Position:");
                ImGui::TableNextColumn();
                ImGui::Text("%.6f", params.optimization.means_lr);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  SH Coeff:");
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", params.optimization.shs_lr);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Opacity:");
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", params.optimization.opacity_lr);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Scaling:");
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", params.optimization.scaling_lr);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Rotation:");
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", params.optimization.rotation_lr);

                // Refinement section
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Refinement:");
                ImGui::TableNextColumn();

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Refine Every:");
                ImGui::TableNextColumn();
                ImGui::Text("%zu", params.optimization.refine_every);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Start Refine:");
                ImGui::TableNextColumn();
                ImGui::Text("%zu", params.optimization.start_refine);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Stop Refine:");
                ImGui::TableNextColumn();
                ImGui::Text("%zu", params.optimization.stop_refine);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Gradient Thr:");
                ImGui::TableNextColumn();
                ImGui::Text("%.6f", params.optimization.grad_threshold);

                if (params.optimization.reset_every > 0) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("  Reset Every:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%zu", params.optimization.reset_every);
                }

                // Strategy-specific parameters
                if (params.optimization.strategy == "mcmc") {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Max Gaussians:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", params.optimization.max_cap);
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Active Features - only show if any are enabled
        bool has_active_features = params.optimization.use_bilateral_grid ||
                                   params.optimization.pose_optimization != "none" ||
                                   params.optimization.enable_eval ||
                                   params.optimization.antialiasing ||
                                   params.optimization.gut;

        if (has_active_features && ImGui::TreeNode("Active Features")) {
            if (ImGui::BeginTable("FeaturesTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Feature", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Configuration", ImGuiTableColumnFlags_WidthStretch);

                if (params.optimization.use_bilateral_grid) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Bilateral Grid:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%dx%dx%d (LR: %.4f)",
                                params.optimization.bilateral_grid_X,
                                params.optimization.bilateral_grid_Y,
                                params.optimization.bilateral_grid_W,
                                params.optimization.bilateral_grid_lr);
                }

                if (params.optimization.pose_optimization != "none") {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Pose Optimization:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", params.optimization.pose_optimization.c_str());
                }

                if (params.optimization.enable_eval) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Evaluation:");
                    ImGui::TableNextColumn();
                    if (!params.optimization.eval_steps.empty()) {
                        std::string steps_str = "Steps: ";
                        for (size_t i = 0; i < params.optimization.eval_steps.size(); ++i) {
                            if (i > 0)
                                steps_str += ", ";
                            steps_str += std::to_string(params.optimization.eval_steps[i]);
                        }
                        ImGui::Text("%s", steps_str.c_str());
                    } else {
                        ImGui::Text("Enabled");
                    }
                }

                if (params.optimization.antialiasing) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Antialiasing:");
                    ImGui::TableNextColumn();
                    ImGui::Text("Enabled");
                }

                if (params.optimization.gut) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("GUT Rasterizer:");
                    ImGui::TableNextColumn();
                    ImGui::Text("Enabled");
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Render Settings
        if (ImGui::TreeNode("Render Settings")) {
            if (ImGui::BeginTable("RenderTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Render Mode:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", params.optimization.render_mode.c_str());

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("SH Degree:");
                ImGui::TableNextColumn();
                ImGui::Text("%d", params.optimization.sh_degree);

                if (!params.optimization.save_steps.empty()) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Save Steps:");
                    ImGui::TableNextColumn();
                    std::string steps_str;
                    for (size_t i = 0; i < params.optimization.save_steps.size(); ++i) {
                        if (i > 0)
                            steps_str += ", ";
                        steps_str += std::to_string(params.optimization.save_steps[i]);
                    }
                    ImGui::Text("%s", steps_str.c_str());
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        ImGui::PopStyleVar();
    }

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
            SaveProjectButton(ctx, state);
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
            SaveProjectButton(ctx, state);
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

        // TRAINING PARAMETERS - NOW DIRECTLY BELOW SAVE PROJECT BUTTON
        if (trainer_state == TrainerManager::State::Ready ||
            trainer_state == TrainerManager::State::Completed) {
            ImGui::Separator();
            if (ImGui::CollapsingHeader("Training Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
                DrawTrainingParameters(ctx);
            }
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

        // Render save project file browser
        if (state.show_save_browser) {
            state.save_browser.render(&state.show_save_browser);
        }
    }

} // namespace gs::gui::panels