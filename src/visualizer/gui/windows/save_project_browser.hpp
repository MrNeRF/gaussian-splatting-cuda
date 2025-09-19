/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <functional>
#include <string>
#ifdef WIN32
#include <Shobjidl.h>
#include <windows.h>
#endif

namespace gs::gui {

    class SaveProjectBrowser {
    public:
        SaveProjectBrowser();

        bool render(bool* p_open);
        void setCurrentPath(const std::filesystem::path& path);
#ifdef WIN32
        bool SaveProjectFileDialog(bool* p_open);
#endif

    private:
        std::string current_path_;
        std::string selected_directory_;
        std::string project_dir_name_;
    };

} // namespace gs::gui