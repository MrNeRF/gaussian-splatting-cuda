/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/training_panel.hpp"
#include "core/events.hpp"
#include "gui/panels/parameter_editor.hpp"
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
            // Get parameters from the project since trainer doesn't expose them
            auto project = ctx.viewer->getProject();
            if (!project) {
                return;
            }
            opt_params = project->getOptimizationParams();
            dataset_params = project->getProjectData().data_set_info;
        }

        // Track changes separately for optimization and dataset parameters
        bool opt_params_changed = false;
        bool dataset_params_changed = false;

        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 12.0f);

        // Strategy Selector
        if (can_edit) {
            ImGui::Text("Strategy:");
            ImGui::SameLine();
            const char* strategies[] = {"default", "mcmc"};
            int current_strategy_idx = (opt_params.strategy == "mcmc") ? 1 : 0;
            int original_idx = current_strategy_idx;

            ImGui::PushItemWidth(150);
            if (ImGui::Combo("##strategy", &current_strategy_idx, strategies, IM_ARRAYSIZE(strategies))) {
                if (current_strategy_idx != original_idx) {
                    std::string new_strategy = strategies[current_strategy_idx];

                    // Load the new strategy's base parameters
                    auto new_params_result = param::OptimizationParameters::from_strategy(new_strategy);
                    if (new_params_result) {
                        // Get current parameters as JSON to preserve user edits
                        nlohmann::json current_overrides = opt_params.params.common();
                        for (const auto& [key, value] : opt_params.params.specific().items()) {
                            current_overrides[key] = value;
                        }

                        // Apply current values as overrides to new strategy
                        opt_params = *new_params_result;
                        opt_params.params = opt_params.params.with_overrides(current_overrides);
                        opt_params_changed = true;
                    }
                }
            }
            ImGui::PopItemWidth();
            ImGui::Separator();
        } else {
            ImGui::Text("Strategy: %s", opt_params.strategy.c_str());
            ImGui::Separator();
        }

        // Create parameter editor
        ParameterEditor editor(opt_params, can_edit);

        // Dataset Parameters
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

                // Resize Factor - EDITABLE
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Resize Factor:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    // Available options
                    static const int resize_options[] = {1, 2, 4, 8};
                    static const char* resize_labels[] = {"1", "2", "4", "8"};
                    static int current_index = 0; // default is 1
                    int array_size = IM_ARRAYSIZE(resize_labels);
                    // Set current_index to current value, if needed
                    for (int i = 0; i < array_size; ++i) {
                        if (dataset_params.resize_factor == resize_options[i]) {
                            current_index = i;
                        }
                    }

                    // Draw combo
                    if (ImGui::Combo("##resize_factor", &current_index, resize_labels, array_size)) {
                        dataset_params.resize_factor = resize_options[current_index];
                        dataset_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", dataset_params.resize_factor);
                }

                // Test Every - EDITABLE (only shown if evaluation is enabled)
                if (opt_params.enable_eval()) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Test Every:");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::PushItemWidth(-1);
                        if (ImGui::InputInt("##test_every", &dataset_params.test_every, 100, 500)) {
                            if (dataset_params.test_every > 0 && dataset_params.test_every <= 10000) {
                                dataset_params_changed = true;
                            }
                        }
                        ImGui::PopItemWidth();
                    } else {
                        ImGui::Text("%d", dataset_params.test_every);
                    }
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

        // Core Training Parameters
        if (ImGui::TreeNode("Core Training")) {
            static const char* render_modes[] = {"RGB", "D", "ED", "RGB_D", "RGB_ED"};

            editor.BeginSection("CoreTable")
                .Row("Iterations:")
                .Int("##iterations", "iterations", 30000, 1000, 5000, 1, 1000000)
                .Row("SH Degree:")
                .SliderInt("##sh_degree", "sh_degree", 3, 1, 3)
                .Row("SH Interval:")
                .Int("##sh_degree_interval", "sh_degree_interval", 1000, 100, 500, 1)
                .Row("Lambda DSSIM:")
                .Slider("##lambda_dssim", "lambda_dssim", 0.2f, 0.0f, 1.0f)
                .Row("Render Mode:")
                .Combo("##render_mode", "render_mode", render_modes, 5, "RGB");
            editor.EndSection();
            ImGui::TreePop();
        }

        // Learning Rates
        if (ImGui::TreeNode("Learning Rates")) {
            editor.BeginSection("LRTable")
                .Row("Position:")
                .Float("##means_lr", "means_lr", 0.00016f, 0.000001f, 0.00001f, "%.6f")
                .Row("SH Coeff:")
                .Float("##shs_lr", "shs_lr", 0.0025f, 0.0001f, 0.001f, "%.4f")
                .Row("Opacity:")
                .Float("##opacity_lr", "opacity_lr", 0.05f, 0.001f, 0.01f, "%.4f")
                .Row("Scaling:")
                .Float("##scaling_lr", "scaling_lr", 0.005f, 0.0001f, 0.001f, "%.4f")
                .Row("Rotation:")
                .Float("##rotation_lr", "rotation_lr", 0.001f, 0.0001f, 0.001f, "%.4f");
            editor.EndSection();
            ImGui::TreePop();
        }

        // Refinement Parameters
        if (ImGui::TreeNode("Refinement")) {
            editor.BeginSection("RefineTable")
                .Row("Refine Every:")
                .Int("##refine_every", "refine_every", 100, 10, 100, 1)
                .Row("Start Refine:")
                .Int("##start_refine", "start_refine", 500, 100, 500, 0)
                .Row("Stop Refine:")
                .Int("##stop_refine", "stop_refine", 15000, 1000, 5000, 0)
                .Row("Gradient Thr:")
                .Float("##grad_threshold", "grad_threshold", 0.0002f, 0.000001f, 0.00001f, "%.6f")
                .Row("Max Gaussians:")
                .Int("##max_cap", "max_cap", 1000000, 10000, 100000, 1);
            editor.EndSection();
            ImGui::TreePop();
        }

        // Regularization Parameters
        if (ImGui::TreeNode("Regularization")) {
            editor.BeginSection("RegTable")
                .Row("Opacity Reg:")
                .Float("##opacity_reg", "opacity_reg", 0.0f, 0.001f, 0.01f, "%.4f", 0.0f)
                .Row("Scale Reg:")
                .Float("##scale_reg", "scale_reg", 0.0f, 0.001f, 0.01f, "%.4f", 0.0f)
                .Row("Min Opacity:")
                .Float("##min_opacity", "min_opacity", 0.005f, 0.001f, 0.01f, "%.4f", 0.0f);
            editor.EndSection();
            ImGui::TreePop();
        }

        // Initialization Parameters
        if (ImGui::TreeNode("Initialization")) {
            bool random_init = opt_params.params.get<bool>("random", false);

            editor.BeginSection("InitTable")
                .Row("Init Opacity:")
                .Slider("##init_opacity", "init_opacity", 0.1f, 0.0f, 1.0f)
                .Row("Init Scaling:")
                .Float("##init_scaling", "init_scaling", 1.0f, 0.01f, 0.1f, "%.3f", 0.0f)
                .Row("Random Init:")
                .Bool("##random", "random", false);

            if (random_init) {
                editor.Row("  Num Points:").Int("##init_num_pts", "init_num_pts", 100000, 1000, 10000, 1).Row("  Extent:").Float("##init_extent", "init_extent", 3.0f, 0.1f, 0.5f, "%.2f", 0.0f);
            }

            editor.EndSection();
            ImGui::TreePop();
        }

        // Strategy-Specific Parameters
        if (opt_params.strategy == "default" && ImGui::TreeNode("Default Strategy")) {
            editor.BeginSection("DefaultTable")
                .Row("Prune Opacity:")
                .Float("##prune_opacity", "prune_opacity", 0.005f, 0.001f, 0.01f, "%.4f")
                .Row("Grow Scale 3D:")
                .Float("##grow_scale3d", "grow_scale3d", 0.01f, 0.001f, 0.01f, "%.4f")
                .Row("Grow Scale 2D:")
                .Float("##grow_scale2d", "grow_scale2d", 0.05f, 0.001f, 0.01f, "%.4f")
                .Row("Prune Scale 3D:")
                .Float("##prune_scale3d", "prune_scale3d", 0.1f, 0.01f, 0.1f, "%.3f")
                .Row("Prune Scale 2D:")
                .Float("##prune_scale2d", "prune_scale2d", 0.15f, 0.01f, 0.1f, "%.3f");

            // Reset Every special case
            size_t reset_every = opt_params.params.get<size_t>("reset_every", 3000);
            if (can_edit) {
                editor.Row("Reset Every:");
                ImGui::PushItemWidth(-1);
                int reset_every_int = static_cast<int>(reset_every);
                if (ImGui::InputInt("##reset_every", &reset_every_int, 100, 1000)) {
                    if (reset_every_int >= 0) {
                        nlohmann::json overrides;
                        overrides["reset_every"] = reset_every_int;
                        opt_params.params = opt_params.params.with_overrides(overrides);
                        opt_params_changed = true;
                    }
                }
                ImGui::PopItemWidth();
            } else {
                editor.Row("Reset Every:");
                if (reset_every > 0) {
                    ImGui::Text("%zu", reset_every);
                } else {
                    ImGui::Text("Disabled");
                }
            }

            editor.Row("Pause After Reset:").Int("##pause_refine_after_reset", "pause_refine_after_reset", 0, 10, 100, 0).Row("Revised Opacity:").Bool("##revised_opacity", "revised_opacity", false);

            editor.EndSection();
            ImGui::TreePop();
        }

        // Advanced Features
        if (ImGui::TreeNode("Advanced Features")) {
            static const char* pose_opts[] = {"none", "direct", "mlp"};
            bool enable_sparsity = opt_params.enable_sparsity();

            editor.BeginSection("AdvancedTable")
                .Row("Enable Eval:")
                .Bool("##enable_eval", "enable_eval", false)
                .Row("Pose Opt:")
                .Combo("##pose_optimization", "pose_optimization", pose_opts, 3, "none")
                .Row("Bilateral Grid:")
                .Bool("##use_bilateral_grid", "use_bilateral_grid", false)
                .Row("Antialiasing:")
                .Bool("##antialiasing", "antialiasing", false)
                .Row("GUT Mode:")
                .Bool("##gut", "gut", false)
                .Row("Enable Sparsity:")
                .Bool("##enable_sparsity", "enable_sparsity", false);

            if (enable_sparsity) {
                editor.Row("  Sparsify Steps:").Int("##sparsify_steps", "sparsify_steps", 15000, 1000, 5000, 1).Row("  Init Rho:").Float("##init_rho", "init_rho", 0.0005f, 0.0001f, 0.001f, "%.6f", 0.0f).Row("  Prune Ratio:").Slider("##prune_ratio", "prune_ratio", 0.6f, 0.0f, 1.0f);
            }

            editor.Row("Save SOG:").Bool("##save_sog", "save_sog", false);
            editor.EndSection();
            ImGui::TreePop();
        }

        // Save Steps
        if (ImGui::TreeNode("Save Steps")) {
            if (can_edit) {
                // Add new save step
                static int new_step = 1000;
                ImGui::InputInt("New Step", &new_step, 100, 1000);
                ImGui::SameLine();
                if (ImGui::Button("Add")) {
                    auto save_steps = opt_params.save_steps();
                    if (new_step > 0 && std::find(save_steps.begin(),
                                                  save_steps.end(),
                                                  new_step) == save_steps.end()) {
                        save_steps.push_back(new_step);
                        std::sort(save_steps.begin(), save_steps.end());

                        // Convert to JSON array and update
                        nlohmann::json save_json = nlohmann::json::array();
                        for (auto s : save_steps) {
                            save_json.push_back(s);
                        }
                        nlohmann::json overrides;
                        overrides["save_steps"] = save_json;
                        opt_params.params = opt_params.params.with_overrides(overrides);
                        opt_params_changed = true;
                    }
                }

                ImGui::Separator();

                // List existing save steps with remove buttons
                auto save_steps = opt_params.save_steps();
                for (size_t i = 0; i < save_steps.size(); ++i) {
                    ImGui::PushID(static_cast<int>(i));

                    int step = static_cast<int>(save_steps[i]);
                    ImGui::SetNextItemWidth(100);
                    if (ImGui::InputInt("##step", &step, 0, 0)) {
                        if (step > 0) {
                            save_steps[i] = static_cast<size_t>(step);
                            std::sort(save_steps.begin(), save_steps.end());

                            // Convert to JSON array and update
                            nlohmann::json save_json = nlohmann::json::array();
                            for (auto s : save_steps) {
                                save_json.push_back(s);
                            }
                            nlohmann::json overrides;
                            overrides["save_steps"] = save_json;
                            opt_params.params = opt_params.params.with_overrides(overrides);
                            opt_params_changed = true;
                        }
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Remove")) {
                        save_steps.erase(save_steps.begin() + i);

                        // Convert to JSON array and update
                        nlohmann::json save_json = nlohmann::json::array();
                        for (auto s : save_steps) {
                            save_json.push_back(s);
                        }
                        nlohmann::json overrides;
                        overrides["save_steps"] = save_json;
                        opt_params.params = opt_params.params.with_overrides(overrides);
                        opt_params_changed = true;
                    }

                    ImGui::PopID();
                }

                if (save_steps.empty()) {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No save steps configured");
                }
            } else {
                // Read-only display
                auto save_steps = opt_params.save_steps();
                if (!save_steps.empty()) {
                    std::string steps_str;
                    for (size_t i = 0; i < save_steps.size(); ++i) {
                        if (i > 0)
                            steps_str += ", ";
                        steps_str += std::to_string(save_steps[i]);
                    }
                    ImGui::Text("%s", steps_str.c_str());
                } else {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No save steps");
                }
            }
            ImGui::TreePop();
        }

        // Eval Steps
        if (opt_params.enable_eval() && ImGui::TreeNode("Eval Steps")) {
            if (can_edit) {
                // Add new eval step
                static int new_eval_step = 1000;
                ImGui::InputInt("New Step", &new_eval_step, 100, 1000);
                ImGui::SameLine();
                if (ImGui::Button("Add")) {
                    auto eval_steps = opt_params.eval_steps();
                    if (new_eval_step > 0 && std::find(eval_steps.begin(),
                                                       eval_steps.end(),
                                                       new_eval_step) == eval_steps.end()) {
                        eval_steps.push_back(new_eval_step);
                        std::sort(eval_steps.begin(), eval_steps.end());

                        // Convert to JSON array and update
                        nlohmann::json eval_json = nlohmann::json::array();
                        for (auto s : eval_steps) {
                            eval_json.push_back(s);
                        }
                        nlohmann::json overrides;
                        overrides["eval_steps"] = eval_json;
                        opt_params.params = opt_params.params.with_overrides(overrides);
                        opt_params_changed = true;
                    }
                }

                ImGui::Separator();

                // List existing eval steps with remove buttons
                auto eval_steps = opt_params.eval_steps();
                for (size_t i = 0; i < eval_steps.size(); ++i) {
                    ImGui::PushID(static_cast<int>(i));

                    int step = static_cast<int>(eval_steps[i]);
                    ImGui::SetNextItemWidth(100);
                    if (ImGui::InputInt("##step", &step, 0, 0)) {
                        if (step > 0) {
                            eval_steps[i] = static_cast<size_t>(step);
                            std::sort(eval_steps.begin(), eval_steps.end());

                            // Convert to JSON array and update
                            nlohmann::json eval_json = nlohmann::json::array();
                            for (auto s : eval_steps) {
                                eval_json.push_back(s);
                            }
                            nlohmann::json overrides;
                            overrides["eval_steps"] = eval_json;
                            opt_params.params = opt_params.params.with_overrides(overrides);
                            opt_params_changed = true;
                        }
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Remove")) {
                        eval_steps.erase(eval_steps.begin() + i);

                        // Convert to JSON array and update
                        nlohmann::json eval_json = nlohmann::json::array();
                        for (auto s : eval_steps) {
                            eval_json.push_back(s);
                        }
                        nlohmann::json overrides;
                        overrides["eval_steps"] = eval_json;
                        opt_params.params = opt_params.params.with_overrides(overrides);
                        opt_params_changed = true;
                    }

                    ImGui::PopID();
                }

                if (eval_steps.empty()) {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No eval steps configured");
                }
            } else {
                // Read-only display
                auto eval_steps = opt_params.eval_steps();
                if (!eval_steps.empty()) {
                    std::string steps_str;
                    for (size_t i = 0; i < eval_steps.size(); ++i) {
                        if (i > 0)
                            steps_str += ", ";
                        steps_str += std::to_string(eval_steps[i]);
                    }
                    ImGui::Text("%s", steps_str.c_str());
                } else {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No eval steps");
                }
            }
            ImGui::TreePop();
        }

        // Check if editor had any changes
        opt_params_changed |= editor.changed();

        // Apply changes if any were made and we can edit
        if ((opt_params_changed || dataset_params_changed) && can_edit) {
            // Update optimization parameters if they changed
            if (opt_params_changed) {
                project->setOptimizationParams(opt_params);
            }

            // Update dataset parameters if they changed
            if (dataset_params_changed) {
                auto project_data = project->getProjectData();

                // Only update the fields from DatasetConfig that we allow editing
                project_data.data_set_info.resize_factor = dataset_params.resize_factor;
                project_data.data_set_info.test_every = dataset_params.test_every;

                // Set the updated project data back
                project->setProjectData(project_data);
            }

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

            if (current_iteration > 0) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.5f, 0.7f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.6f, 0.8f, 1.0f));
                if (ImGui::Button("Reset Training", ImVec2(-1, 0))) {
                    trainer_manager->resetTraining();
                }
                ImGui::PopStyleColor(2);
            }

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

            // Reset button
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.5f, 0.7f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.6f, 0.8f, 1.0f));
            if (ImGui::Button("Reset Training", ImVec2(-1, 0))) {
                trainer_manager->resetTraining();
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

            // Reset button for completed state
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.5f, 0.7f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.6f, 0.8f, 1.0f));
            if (ImGui::Button("Reset for New Training", ImVec2(-1, 0))) {
                trainer_manager->resetTraining();
            }
            ImGui::PopStyleColor(2);

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
