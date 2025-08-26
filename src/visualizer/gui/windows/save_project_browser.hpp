#pragma once

#include <filesystem>
#include <functional>
#include <string>

namespace gs::gui {

    class SaveProjectBrowser {
    public:
        SaveProjectBrowser();

        void render(bool* p_open);
        void setCurrentPath(const std::filesystem::path& path);

    private:
        std::string current_path_;
        std::string selected_directory_;
        std::string project_dir_name_;
    };

} // namespace gs::gui