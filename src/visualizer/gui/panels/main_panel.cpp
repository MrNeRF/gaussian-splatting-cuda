#include "gui/panels/main_panel.hpp"
#include "core/events.hpp"
#include "gui/panels/crop_box_panel.hpp"
#include "gui/panels/training_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <format>
#include <imgui.h>

namespace gs::gui::panels {

    void DrawMainPanel(const UIContext& ctx) {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));

        ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar |
                                 ImGuiWindowFlags_NoResize;

        if (ImGui::Begin("Rendering Setting", nullptr, flags)) {
            ImGui::SetWindowSize(ImVec2(300, 0));

            DrawWindowControls(ctx);
            ImGui::Separator();

            widgets::DrawModeStatus(ctx);
            ImGui::Separator();

            DrawRenderingSettings(ctx);
            ImGui::Separator();

            if (ctx.viewer->getTrainer()) {
                DrawTrainingControls(ctx);
                ImGui::Separator();
            }

            DrawProgressInfo(ctx);
            ImGui::Separator();

            DrawCropBoxControls(ctx);
        }
        ImGui::End();

        ImGui::PopStyleColor();
    }

    void DrawWindowControls(const UIContext& ctx) {
        if (ImGui::Button("Open Scripting Console", ImVec2(-1, 0))) {
            (*ctx.window_states)["console"] = true;
        }

        if (ImGui::Button("Open Camera Controls", ImVec2(-1, 0))) {
            (*ctx.window_states)["camera_controls"] = true;
        }

        ImGui::Separator();
        ImGui::Text("Windows");
        ImGui::Checkbox("Scripting Console", &(*ctx.window_states)["console"]);
        ImGui::Checkbox("Camera Controls", &(*ctx.window_states)["camera_controls"]);
        ImGui::Checkbox("Scene Panel", &(*ctx.window_states)["scene_panel"]);
    }

    void DrawRenderingSettings(const UIContext& ctx) {
        auto config = ctx.viewer->getRenderingConfig();
        if (!config)
            return;

        ImGui::Text("Rendering Settings");
        ImGui::Separator();

        float old_scale = config->scaling_modifier;
        if (widgets::SliderWithReset("Scale", &config->scaling_modifier, 0.01f, 3.0f, 1.0f)) {
            ctx.event_bus->publish(RenderingSettingsChangedEvent{
                std::nullopt, config->scaling_modifier, std::nullopt});
        }

        float old_fov = config->fov;
        if (widgets::SliderWithReset("FoV", &config->fov, 45.0f, 120.0f, 75.0f)) {
            ctx.event_bus->publish(RenderingSettingsChangedEvent{
                config->fov, std::nullopt, std::nullopt});
        }

#ifdef CUDA_GL_INTEROP_ENABLED
        ImGui::Text("Render Mode: GPU Direct (Interop)");
#else
        ImGui::Text("Render Mode: CPU Copy");
#endif
    }

    void DrawProgressInfo(const UIContext& ctx) {
        auto info = ctx.viewer->getTrainingInfo();

        int current_iter = info->curr_iterations_.load();
        int total_iter = info->total_iterations_.load();
        int num_splats = info->num_splats_.load();

        std::vector<float> loss_data;
        {
            std::lock_guard<std::mutex> lock(info->loss_buffer_mutex_);
            loss_data.assign(info->loss_buffer_.begin(), info->loss_buffer_.end());
        }

        float fraction = total_iter > 0 ? float(current_iter) / float(total_iter) : 0.0f;
        char overlay_text[64];
        std::snprintf(overlay_text, sizeof(overlay_text), "%d / %d", current_iter, total_iter);

        widgets::DrawProgressBar(fraction, overlay_text);

        if (loss_data.size() > 0) {
            auto [min_it, max_it] = std::minmax_element(loss_data.begin(), loss_data.end());
            float min_val = *min_it, max_val = *max_it;

            if (min_val == max_val) {
                min_val -= 1.0f;
                max_val += 1.0f;
            } else {
                float margin = (max_val - min_val) * 0.05f;
                min_val -= margin;
                max_val += margin;
            }

            char loss_label[64];
            std::snprintf(loss_label, sizeof(loss_label), "Loss: %.4f", loss_data.back());

            widgets::DrawLossPlot(loss_data.data(), static_cast<int>(loss_data.size()),
                                  min_val, max_val, loss_label);
        }

        ImGui::Text("num Splats: %d", num_splats);
    }
} // namespace gs::gui::panels
