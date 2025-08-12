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

        // Get current render settings
        auto render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager)
            return;

        auto settings = render_manager->getSettings();
        bool settings_changed = false;

        // Point Cloud Mode checkbox
        if (ImGui::Checkbox("Point Cloud Mode", &settings.point_cloud_mode)) {
            settings_changed = true;
        }

        // Show voxel size slider only when in point cloud mode
        if (settings.point_cloud_mode) {
            if (widgets::SliderWithReset("Voxel Size", &settings.voxel_size, 0.001f, 0.1f, 0.01f)) {
                settings_changed = true;
            }
        }

        // Apply settings changes if any
        if (settings_changed) {
            render_manager->updateSettings(settings);

            // Force a camera update to trigger re-render
            const auto& viewport = ctx.viewer->getViewport();
            events::ui::CameraMove{
                .rotation = viewport.getRotationMatrix(),
                .translation = viewport.getTranslation()}
                .emit();
        }

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

        // Display current FPS (read-only)
        float average_fps = ctx.viewer->getAverageFPS();
        if (average_fps > 0.0f) {
            ImGui::Text("FPS: %.1f", average_fps);
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