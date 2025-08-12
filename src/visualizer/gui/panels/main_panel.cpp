#include "gui/panels/main_panel.hpp"
#include "core/events.hpp"
#include "gui/panels/tools_panel.hpp"
#include "gui/panels/training_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <imgui.h>

namespace gs::gui::panels {

    void DrawMainPanel(const UIContext& ctx) {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));

        // Simplified flags - positioning is handled in GuiManager::render()
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar |
                                 ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_NoTitleBar; // Add this to remove title bar

        if (ImGui::Begin("Rendering Setting", nullptr, flags)) {
            // Add a custom title
            ImGui::Text("Rendering Settings");
            ImGui::Separator();

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

            DrawToolsPanel(ctx);
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

        if (widgets::SliderWithReset("Scale", &config->scaling_modifier, 0.01f, 3.0f, 1.0f)) {
            events::ui::RenderSettingsChanged{
                .fov = std::nullopt,
                .scaling_modifier = config->scaling_modifier,
                .antialiasing = std::nullopt,
                .background_color = std::nullopt}
                .emit();
        }

        if (widgets::SliderWithReset("FoV", &config->fov, 45.0f, 120.0f, 75.0f)) {
            events::ui::RenderSettingsChanged{
                .fov = config->fov,
                .scaling_modifier = std::nullopt,
                .antialiasing = std::nullopt,
                .background_color = std::nullopt}
                .emit();
        }

        // Display current FPS and VSync control on the same line
        float average_fps = ctx.viewer->getAverageFPS();
        if (average_fps > 0.0f) {
            ImGui::Text("FPS: %6.1f", average_fps); // 6 characters total, 1 decimal place

            // Add VSync checkbox on the same line
            ImGui::SameLine();
            ImGui::Spacing();
            ImGui::SameLine();

            // Get current VSync state from viewer
            bool vsync_enabled = ctx.viewer->getVSyncEnabled();

            if (ImGui::Checkbox("VSync", &vsync_enabled)) {
                // Set VSync through the viewer's public interface
                ctx.viewer->setVSync(vsync_enabled);

                // should we create this event ?
                // events::ui::VSyncChanged{
                //     .enabled = vsync_enabled
                // }.emit();
            }

            // Add tooltip
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Toggle Vertical Synchronization\n%s",
                                  vsync_enabled ? "FPS capped to monitor refresh rate"
                                                : "Uncapped FPS");
            }
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