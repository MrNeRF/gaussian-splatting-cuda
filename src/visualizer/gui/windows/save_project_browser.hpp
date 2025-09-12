/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <functional>
#include <string>

namespace gs::gui {

    class SaveProjectBrowser {
    public:
        SaveProjectBrowser();

        bool render(bool* p_open);
        void setCurrentPath(const std::filesystem::path& path);

    private:
        std::string current_path_;
        std::string selected_directory_;
        std::string project_dir_name_;
    };

} // namespace gs::gui