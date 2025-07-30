#include "gui/windows/camera_controls.hpp"
#include <imgui.h>

namespace gs::gui::windows {

    void DrawCameraControls(bool* p_open) {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.15f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

        if (ImGui::Begin("Camera Controls", p_open, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Camera Controls:");
            ImGui::Separator();

            if (ImGui::BeginTable("camera_controls_table", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 400.0f);
                ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Local Translate Camera");
                ImGui::TableNextColumn();
                ImGui::Text("Left Mouse + Drag");

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Local Rotate Camera (Pitch/Yaw)");
                ImGui::TableNextColumn();
                ImGui::Text("Right Mouse + Drag");

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Rotate Around Scene Center");
                ImGui::TableNextColumn();
                ImGui::Text("Middle Mouse + Drag");

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Zoom");
                ImGui::TableNextColumn();
                ImGui::Text("Mouse Scroll");

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Roll Camera");
                ImGui::TableNextColumn();
                ImGui::Text("R + Mouse Scroll");

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Move forward, backward, left and right within the scene");
                ImGui::TableNextColumn();
                ImGui::Text("w, s, a, d keys");

                ImGui::EndTable();
            }

            ImGui::Separator();
        }
        ImGui::End();
        ImGui::PopStyleColor(4);
    }
} // namespace gs::gui::windows
