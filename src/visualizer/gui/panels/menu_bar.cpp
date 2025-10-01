/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/menu_bar.hpp"
#include "core/logger.hpp"
#include <imgui.h>

namespace gs::gui {

    MenuBar::MenuBar() {
        LOG_DEBUG("MenuBar created");
    }

    MenuBar::~MenuBar() {
        // Cleanup handled automatically
    }

    void MenuBar::render() {
        if (ImGui::BeginMainMenuBar()) {
            // File menu
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Import Dataset")) {
                    LOG_DEBUG("Import Dataset clicked");
                    if (on_import_dataset_) {
                        on_import_dataset_();
                    }
                }

                if (ImGui::MenuItem("Open Project")) {
                    LOG_DEBUG("Open Project clicked");
                    if (on_open_project_) {
                        on_open_project_();
                    }
                }

                ImGui::EndMenu();
            }

            // Help menu
            if (ImGui::BeginMenu("Help")) {
                if (ImGui::MenuItem("About")) {
                    LOG_DEBUG("About clicked");
                    show_about_window_ = true;
                }

                if (ImGui::MenuItem("Camera Controls")) {
                    LOG_DEBUG("Camera Controls clicked");
                    show_camera_controls_ = true;
                }

                ImGui::EndMenu();
            }

            ImGui::EndMainMenuBar();
        }
    }

    void MenuBar::renderAboutWindow() {
        if (!show_about_window_) {
            return;
        }

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.15f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

        if (ImGui::Begin("About", &show_about_window_, window_flags)) {
            ImGui::Text("LichtFeld Project but MrNerf");
        }
        ImGui::End();

        ImGui::PopStyleColor(4);
    }

    void MenuBar::renderCameraControlsWindow() {
        if (!show_camera_controls_) {
            return;
        }

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.15f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

        if (ImGui::Begin("Camera Controls", &show_camera_controls_, window_flags)) {
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

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Increase/Decrease wasd speed");
                ImGui::TableNextColumn();
                ImGui::Text("Ctrl + +/-");

                ImGui::EndTable();
            }
            ImGui::Separator();
        }
        ImGui::End();

        ImGui::PopStyleColor(4);
    }

    void MenuBar::setOnImportDataset(std::function<void()> callback) {
        on_import_dataset_ = std::move(callback);
    }

    void MenuBar::setOnOpenProject(std::function<void()> callback) {
        on_open_project_ = std::move(callback);
    }

} // namespace gs::gui