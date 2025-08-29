/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <functional>
#include <string>

namespace gs::gui {

    class FileBrowser {
    public:
        FileBrowser();

        void render(bool* p_open);
        void setOnFileSelected(std::function<void(const std::filesystem::path&, bool)> callback);
        void setCurrentPath(const std::filesystem::path& path);
        void setSelectedPath(const std::filesystem::path& path);

    private:
        std::string current_path_;
        std::string selected_file_;
        std::function<void(const std::filesystem::path&, bool)> on_file_selected_;
    };

} // namespace gs::gui
