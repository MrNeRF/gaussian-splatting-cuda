/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/menu_bar.hpp"
#include "core/logger.hpp"
#include <imgui.h>

#include <cstdlib> // for system()
#ifdef _WIN32
#include <windows.h> // for ShellExecuteA
#endif

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
                if (ImGui::MenuItem("Open Project")) {
                    LOG_DEBUG("Open Project clicked");
                    if (on_open_project_) {
                        on_open_project_();
                    }
                }

                if (ImGui::MenuItem("Import Dataset")) {
                    LOG_DEBUG("Import Dataset clicked");
                    if (on_import_dataset_) {
                        on_import_dataset_();
                    }
                }

                if (ImGui::MenuItem("Import Ply")) {
                    LOG_DEBUG("Import Ply clicked");
                    if (on_import_ply_) {
                        on_import_ply_();
                    }
                }

                ImGui::Separator();
                if (ImGui::MenuItem("Save Project As")) {
                    LOG_DEBUG("Save Project As clicked");
                    if (on_save_project_as_) {
                        on_save_project_as_();
                    }
                }

                // Disable "Save Project" if project is temporary
                if (ImGui::MenuItem("Save Project", nullptr, false, !is_project_temp_)) {
                    LOG_DEBUG("Save Project clicked");
                    if (on_save_project_) {
                        on_save_project_();
                    }
                }

                ImGui::EndMenu();
            }

            // Help menu
            if (ImGui::BeginMenu("Help")) {
                if (ImGui::MenuItem("Getting Started")) {
                    LOG_DEBUG("Getting Started clicked");
                    show_getting_started_ = true;
                }

                if (ImGui::MenuItem("Camera Controls")) {
                    LOG_DEBUG("Camera Controls clicked");
                    show_camera_controls_ = true;
                }

                if (ImGui::MenuItem("About LichtFeld Studio")) {
                    LOG_DEBUG("About clicked");
                    show_about_window_ = true;
                }

                ImGui::EndMenu();
            }

            ImGui::EndMainMenuBar();
        }

        renderGettingStartedWindow();
        renderAboutWindow();
        renderCameraControlsWindow();
    }

    void MenuBar::openURL(const char* url) {
#ifdef _WIN32
        ShellExecuteA(nullptr, "open", url, nullptr, nullptr, SW_SHOWNORMAL);
#elif __APPLE__
        std::string cmd = "open " + std::string(url);
        system(cmd.c_str());
#else
        std::string cmd = "xdg-open " + std::string(url);
        system(cmd.c_str());
#endif
    }

    void MenuBar::renderGettingStartedWindow() {
        if (!show_getting_started_) {
            return;
        }

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize;

        // Initial width (fits nicely, but height will adapt to content)
        ImGui::SetNextWindowSize(ImVec2(650, 0), ImGuiCond_Once);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.15f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

        if (ImGui::Begin("Getting Started", &show_getting_started_, window_flags)) {
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.4f, 1.0f), "Usage");
            ImGui::Separator();

            ImGui::TextWrapped("Introduction Tutorial: dataset preparation");
            ImGui::Spacing();

            // Reality Scan video
            const char* reality_scan_url = "http://www.youtube.com/watch?v=JWmkhTlbDvg";
            ImGui::Bullet();
            ImGui::TextWrapped("Using Reality Scan to create a dataset");
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.8f, 1.0f, 1.0f));
            ImGui::Text("%s", reality_scan_url);
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                if (ImGui::IsItemClicked()) {
                    openURL(reality_scan_url);
                }
            }

            // Colmap tutorial video
            const char* colmap_tutorial_url = "https://www.youtube.com/watch?v=-3TBbukYN00";
            ImGui::Bullet();
            ImGui::TextWrapped("Beginner Tutorial - Using COLMAP to create a dataset for LichtFeld Studio");
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.8f, 1.0f, 1.0f));
            ImGui::Text("%s", colmap_tutorial_url);
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                if (ImGui::IsItemClicked()) {
                    openURL(colmap_tutorial_url);
                }
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // FAQ link
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.4f, 1.0f), "FAQ:");
            const char* faq_url = "https://github.com/MrNeRF/LichtFeld-Studio/blob/master/docs/docs/faq.md";
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.8f, 1.0f, 1.0f));
            ImGui::Text("%s", faq_url);
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                if (ImGui::IsItemClicked()) {
                    openURL(faq_url);
                }
            }
        }
        ImGui::End();

        ImGui::PopStyleColor(4);
    }

    void MenuBar::renderAboutWindow() {
        if (!show_about_window_) {
            return;
        }

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize;

        // Initial width (fits nicely, but height will adapt to content)
        ImGui::SetNextWindowSize(ImVec2(650, 0), ImGuiCond_Once);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.15f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

        std::string version = "v0.1.3";
        const char* repo_url = "https://github.com/MrNeRF/LichtFeld-Studio";

        if (ImGui::Begin("About LichtFeld Studio", &show_about_window_, window_flags)) {
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.4f, 1.0f), "LichtFeld Studio");
            ImGui::Separator();

            ImGui::TextWrapped(
                "LichtFeld Studio is a high-performance C++ and CUDA implementation of 3D Gaussian Splatting, "
                "designed to fuse real and digital content seamlessly. It empowers research and development in "
                "neural rendering, allowing real-time visualization, training, and inspection of neural radiance fields.");

            ImGui::Spacing();
            ImGui::Text("Version: %s", version.c_str());
            ImGui::Spacing();

            ImGui::TextWrapped("Source code, docs, and full project details are hosted on GitHub:");

            // Clickable link
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.8f, 1.0f, 1.0f));
            ImGui::Text("%s", repo_url);
            ImGui::PopStyleColor();

            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                if (ImGui::IsItemClicked()) {
                    openURL(repo_url);
                }
            }

            ImGui::Spacing();
            ImGui::Separator();

            ImGui::Text("Created by Mr Nerf");

            ImGui::Spacing();
            ImGui::Separator();

            ImGui::TextWrapped("This software is licensed under GPLv3. See the LICENSE file for full terms.");
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

    void MenuBar::setOnImportPLY(std::function<void()> callback) {
        on_import_ply_ = std::move(callback);
    }

    void MenuBar::setOnSaveProjectAs(std::function<void()> callback) {
        on_save_project_as_ = std::move(callback);
    }

    void MenuBar::setOnSaveProject(std::function<void()> callback) {
        on_save_project_ = std::move(callback);
    }

} // namespace gs::gui