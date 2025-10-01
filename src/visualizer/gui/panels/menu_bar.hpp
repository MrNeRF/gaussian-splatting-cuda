/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
*
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>

namespace gs::gui {

    class MenuBar {
    public:
        MenuBar();
        ~MenuBar();

        // Render the menu bar
        void render();

        // Callbacks for menu actions
        void setOnImportDataset(std::function<void()> callback);
        void setOnOpenProject(std::function<void()> callback);
        void setOnImportPLY(std::function<void()> callback);

        // Window state management
        bool isAboutWindowOpen() const { return show_about_window_; }
        bool isCameraControlsWindowOpen() const { return show_camera_controls_; }

        void setAboutWindowOpen(bool open) { show_about_window_ = open; }
        void setCameraControlsWindowOpen(bool open) { show_camera_controls_ = open; }

        // Render separate windows (call these in your main render loop)
        void renderAboutWindow();
        void renderCameraControlsWindow();

    private:
        // Callbacks
        std::function<void()> on_import_dataset_;
        std::function<void()> on_open_project_;
        std::function<void()> on_import_ply_;

        // Window states
        bool show_about_window_ = false;
        bool show_camera_controls_ = false;
    };

} // namespace gs::gui