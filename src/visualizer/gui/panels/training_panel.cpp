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

        // Get current state to determine if we can edit
        auto trainer_state = trainer_manager->getState();
        bool can_edit = (trainer_state == TrainerManager::State::Ready);

        // Get project to modify parameters if we're in edit mode
        auto project = ctx.viewer->getProject();
        if (!project) {
            return;
        }

        // Get parameters - either from project (if Ready) or from trainer (if training/completed)
        param::OptimizationParameters opt_params;
        param::DatasetConfig dataset_params;

        if (trainer_state == TrainerManager::State::Ready) {
            // Before training - get from project (editable)
            opt_params = project->getOptimizationParams();
            dataset_params = project->getProjectData().data_set_info;
        } else {
            // During/after training - get from trainer (read-only)
            const auto* trainer = trainer_manager->getTrainer();
            if (!trainer) {
                return;
            }
            const auto& params = trainer->getParams();
            opt_params = params.optimization;
            dataset_params = params.dataset;
        }

        bool params_changed = false;

        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 12.0f);

        // Dataset Parameters (always read-only)
        if (ImGui::TreeNode("Dataset")) {
            if (ImGui::BeginTable("DatasetTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Path:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", dataset_params.data_path.filename().string().c_str());

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Images:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", dataset_params.images.c_str());

                if (dataset_params.resize_factor != -1) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Resize Factor:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", dataset_params.resize_factor);
                }

                if (opt_params.enable_eval) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Test Every:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", dataset_params.test_every);
                }

                if (!dataset_params.output_path.empty()) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Output:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", dataset_params.output_path.filename().string().c_str());
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Optimization Parameters (editable when Ready)
        if (ImGui::TreeNode("Optimization")) {
            if (ImGui::BeginTable("OptimizationTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                // Iterations - EDITABLE
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Iterations:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int iterations = static_cast<int>(opt_params.iterations);
                    if (ImGui::InputInt("##iterations", &iterations, 1000, 5000)) {
                        if (iterations > 0 && iterations <= 1000000) {
                            opt_params.iterations = static_cast<size_t>(iterations);
                            params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%zu", opt_params.iterations);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Strategy:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", opt_params.strategy.c_str());

                // Learning Rates section
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Learning Rates:");
                ImGui::TableNextColumn();

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Position:");
                ImGui::TableNextColumn();
                ImGui::Text("%.6f", opt_params.means_lr);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  SH Coeff:");
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", opt_params.shs_lr);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Opacity:");
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", opt_params.opacity_lr);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Scaling:");
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", opt_params.scaling_lr);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Rotation:");
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", opt_params.rotation_lr);

                // Refinement section
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Refinement:");
                ImGui::TableNextColumn();

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Refine Every:");
                ImGui::TableNextColumn();
                ImGui::Text("%zu", opt_params.refine_every);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Start Refine:");
                ImGui::TableNextColumn();
                ImGui::Text("%zu", opt_params.start_refine);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Stop Refine:");
                ImGui::TableNextColumn();
                ImGui::Text("%zu", opt_params.stop_refine);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Gradient Thr:");
                ImGui::TableNextColumn();
                ImGui::Text("%.6f", opt_params.grad_threshold);

                if (opt_params.reset_every > 0) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("  Reset Every:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%zu", opt_params.reset_every);
                }

                // Strategy-specific parameters
                if (opt_params.strategy == "mcmc") {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Max Gaussians:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", opt_params.max_cap);
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Active Features - only show if any are enabled
        bool has_active_features = opt_params.use_bilateral_grid ||
                                   opt_params.pose_optimization != "none" ||
                                   opt_params.enable_eval ||
                                   opt_params.antialiasing ||
                                   opt_params.gut;

        if (has_active_features && ImGui::TreeNode("Active Features")) {
            if (ImGui::BeginTable("FeaturesTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Feature", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Configuration", ImGuiTableColumnFlags_WidthStretch);

                if (opt_params.use_bilateral_grid) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Bilateral Grid:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%dx%dx%d (LR: %.4f)",
                                opt_params.bilateral_grid_X,
                                opt_params.bilateral_grid_Y,
                                opt_params.bilateral_grid_W,
                                opt_params.bilateral_grid_lr);
                }

                if (opt_params.pose_optimization != "none") {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Pose Optimization:");
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", opt_params.pose_optimization.c_str());
                }

                if (opt_params.enable_eval) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Evaluation:");
                    ImGui::TableNextColumn();
                    if (!opt_params.eval_steps.empty()) {
                        std::string steps_str = "Steps: ";
                        for (size_t i = 0; i < opt_params.eval_steps.size(); ++i) {
                            if (i > 0)
                                steps_str += ", ";
                            steps_str += std::to_string(opt_params.eval_steps[i]);
                        }
                        ImGui::Text("%s", steps_str.c_str());
                    } else {
                        ImGui::Text("Enabled");
                    }
                }

                if (opt_params.antialiasing) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Antialiasing:");
                    ImGui::TableNextColumn();
                    ImGui::Text("Enabled");
                }

                if (opt_params.gut) {
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
                ImGui::Text("%s", opt_params.render_mode.c_str());

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("SH Degree:");
                ImGui::TableNextColumn();
                ImGui::Text("%d", opt_params.sh_degree);

                if (!opt_params.save_steps.empty()) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Save Steps:");
                    ImGui::TableNextColumn();
                    std::string steps_str;
                    for (size_t i = 0; i < opt_params.save_steps.size(); ++i) {
                        if (i > 0)
                            steps_str += ", ";
                        steps_str += std::to_string(opt_params.save_steps[i]);
                    }
                    ImGui::Text("%s", steps_str.c_str());
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Apply changes if any were made and we can edit
        if (params_changed && can_edit) {
            project->setOptimizationParams(opt_params);
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f),
                "Parameters updated - will be applied when training starts");
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

        ImGui::Separator();
        if (ImGui::CollapsingHeader("Training Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            DrawTrainingParameters(ctx);
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