#include "gui/ui_widgets.hpp"
#include "core/event_response_handler.hpp"
#include "core/events.hpp"
#include "visualizer_impl.hpp"
#include <cuda_runtime.h>
#include <format>
#include <imgui.h>
#include <iomanip>
#include <sstream>

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
        EventResponseHandler<QuerySceneModeRequest, QuerySceneModeResponse> handler(ctx.event_bus);
        auto response = handler.querySync(QuerySceneModeRequest{});

        if (response) {
            switch (response->mode) {
            case QuerySceneModeResponse::Mode::Empty:
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No data loaded");
                ImGui::Text("Use File Browser to load:");
                ImGui::BulletText("PLY file for viewing");
                ImGui::BulletText("Dataset for training");
                break;

            case QuerySceneModeResponse::Mode::Viewing:
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "PLY Viewer Mode");
                if (ctx.viewer->getCurrentPLYPath().has_filename()) {
                    ImGui::Text("File: %s", ctx.viewer->getCurrentPLYPath().filename().string().c_str());
                }
                break;

            case QuerySceneModeResponse::Mode::Training:
                ImGui::TextColored(ImVec4(0.2f, 0.5f, 0.8f, 1.0f), "Training Mode");
                if (ctx.viewer->getCurrentDatasetPath().has_filename()) {
                    ImGui::Text("Dataset: %s", ctx.viewer->getCurrentDatasetPath().filename().string().c_str());
                }
                break;
            }
        }
    }

    const char* GetTrainerStateString(int state) {
        switch (state) {
        case 0: return "Idle";
        case 1: return "Ready";
        case 2: return "Running";
        case 3: return "Paused";
        case 4: return "Stopping";
        case 5: return "Completed";
        case 6: return "Error";
        default: return "Unknown";
        }
    }

    std::string executeConsoleCommand(const std::string& command,
                                      visualizer::VisualizerImpl* viewer,
                                      std::shared_ptr<EventBus> event_bus) {
        std::ostringstream result;

        if (command.empty()) {
            return "";
        }

        if (command == "help" || command == "h") {
            result << "Available commands:\n";
            result << "  help, h - Show this help\n";
            result << "  clear - Clear console\n";
            result << "  status - Show training status\n";
            result << "  model_info - Show model information\n";
            result << "  tensor_info <name> - Show tensor information\n";
            result << "  gpu_info - Show GPU information\n";
            return result.str();
        }

        if (command == "clear") {
            return "";
        }

        if (command == "status") {
            EventResponseHandler<QueryTrainerStateRequest, QueryTrainerStateResponse> handler(event_bus);
            auto response = handler.querySync(QueryTrainerStateRequest{});

            if (response) {
                result << "Training Status:\n";
                result << "  State: " << GetTrainerStateString(static_cast<int>(response->state)) << "\n";
                result << "  Current Iteration: " << response->current_iteration << "\n";
                result << "  Current Loss: " << std::fixed << std::setprecision(6) << response->current_loss;
                if (response->error_message) {
                    result << "\n  Error: " << *response->error_message;
                }
            } else {
                result << "No trainer available (viewer mode)";
            }
            return result.str();
        }

        if (command == "model_info") {
            EventResponseHandler<QuerySceneStateRequest, QuerySceneStateResponse> stateHandler(event_bus);
            auto stateResponse = stateHandler.querySync(QuerySceneStateRequest{});

            if (stateResponse && stateResponse->has_model) {
                result << "Scene Information:\n";
                result << "  Type: " << [&]() {
                    switch (stateResponse->type) {
                    case QuerySceneStateResponse::SceneType::None: return "None";
                    case QuerySceneStateResponse::SceneType::PLY: return "PLY";
                    case QuerySceneStateResponse::SceneType::Dataset: return "Dataset";
                    default: return "Unknown";
                    }
                }() << "\n";
                result << "  Source: " << stateResponse->source_path.filename().string() << "\n";
                result << "  Number of Gaussians: " << stateResponse->num_gaussians << "\n";

                if (stateResponse->is_training) {
                    result << "  Training Iteration: " << stateResponse->training_iteration.value_or(0) << "\n";
                }

                EventResponseHandler<QueryModelInfoRequest, QueryModelInfoResponse> modelHandler(event_bus);
                auto modelResponse = modelHandler.querySync(QueryModelInfoRequest{});

                if (modelResponse) {
                    result << "  SH Degree: " << modelResponse->sh_degree << "\n";
                    result << "  Scene Scale: " << modelResponse->scene_scale << "\n";
                }
            } else {
                result << "No scene loaded";
            }
            return result.str();
        }

        if (command == "gpu_info") {
            size_t free_byte, total_byte;
            cudaDeviceSynchronize();
            cudaMemGetInfo(&free_byte, &total_byte);

            double free_gb = free_byte / 1024.0 / 1024.0 / 1024.0;
            double total_gb = total_byte / 1024.0 / 1024.0 / 1024.0;
            double used_gb = total_gb - free_gb;

            result << "GPU Memory Info:\n";
            result << "  Total: " << std::fixed << std::setprecision(2) << total_gb << " GB\n";
            result << "  Used: " << used_gb << " GB\n";
            result << "  Free: " << free_gb << " GB\n";
            result << "  Usage: " << std::setprecision(1) << (used_gb / total_gb * 100.0) << "%";
            return result.str();
        }

        if (command.substr(0, 11) == "tensor_info") {
            if (command.length() <= 12) {
                return "Usage: tensor_info <tensor_name>\nAvailable: means, scaling, rotation, shs, opacity";
            }

            std::string tensor_name = command.substr(12);
            auto* scene_manager = viewer->getSceneManager();

            if (!scene_manager || !scene_manager->hasScene()) {
                return "No model available";
            }

            auto* model = scene_manager->getScene()->getMutableModel();
            if (!model) {
                return "Model not available";
            }

            torch::Tensor tensor;
            if (tensor_name == "means" || tensor_name == "positions") {
                tensor = model->get_means();
            } else if (tensor_name == "scales" || tensor_name == "scaling") {
                tensor = model->get_scaling();
            } else if (tensor_name == "rotations" || tensor_name == "rotation" || tensor_name == "quats") {
                tensor = model->get_rotation();
            } else if (tensor_name == "features" || tensor_name == "colors" || tensor_name == "shs") {
                tensor = model->get_shs();
            } else if (tensor_name == "opacities" || tensor_name == "opacity") {
                tensor = model->get_opacity();
            } else {
                return "Unknown tensor: " + tensor_name + "\nAvailable: means, scaling, rotation, shs, opacity";
            }

            std::ostringstream oss;
            oss << "Tensor '" << tensor_name << "' info:\n";
            oss << "  Shape: [";
            for (int i = 0; i < tensor.dim(); i++) {
                if (i > 0)
                    oss << ", ";
                oss << tensor.size(i);
            }
            oss << "]\n";
            oss << "  Device: " << tensor.device() << "\n";
            oss << "  Dtype: " << tensor.dtype() << "\n";
            oss << "  Requires grad: " << (tensor.requires_grad() ? "Yes" : "No") << "\n";

            try {
                auto cpu_tensor = tensor.cpu();
                auto flat = cpu_tensor.flatten();
                if (flat.numel() > 0) {
                    oss << "  Min: " << torch::min(flat).item<float>() << "\n";
                    oss << "  Max: " << torch::max(flat).item<float>() << "\n";
                    oss << "  Mean: " << torch::mean(flat).item<float>() << "\n";
                    oss << "  Std: " << torch::std(flat).item<float>();
                }
            } catch (...) {
                oss << "  (Statistics unavailable)";
            }

            return oss.str();
        }

        return "Unknown command: '" + command + "'. Type 'help' for available commands.";
    }
} // namespace gs::gui::widgets
