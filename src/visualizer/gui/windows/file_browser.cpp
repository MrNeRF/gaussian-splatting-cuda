#include "gui/windows/file_browser.hpp"
#include <algorithm>
#include <imgui.h>
#include <print>

namespace gs::gui {

    FileBrowser::FileBrowser() {
        current_path_ = std::filesystem::current_path().string();
    }

    void FileBrowser::render(bool* p_open) {
        ImGui::SetNextWindowSize(ImVec2(700, 450), ImGuiCond_FirstUseEver);

        // Add NoDocking flag
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.15f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));

        if (!ImGui::Begin("File Browser", p_open, window_flags)) {
            ImGui::End();
            ImGui::PopStyleColor(2);
            return;
        }

        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("Quick Access")) {
                if (ImGui::MenuItem("Current Directory")) {
                    current_path_ = std::filesystem::current_path().string();
                }
                if (ImGui::MenuItem("Home")) {
                    current_path_ = std::filesystem::path(std::getenv("HOME") ? std::getenv("HOME") : "/").string();
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        ImGui::Text("Current Path: %s", current_path_.c_str());
        ImGui::Separator();

        if (ImGui::BeginChild("FileList", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 2), true)) {
            std::filesystem::path current_path(current_path_);

            if (current_path.has_parent_path()) {
                if (ImGui::Selectable("../", false, ImGuiSelectableFlags_DontClosePopups)) {
                    current_path_ = current_path.parent_path().string();
                    selected_file_.clear();
                }
            }

            std::vector<std::filesystem::directory_entry> dirs;
            std::vector<std::filesystem::directory_entry> files;

            try {
                for (const auto& entry : std::filesystem::directory_iterator(current_path)) {
                    if (entry.is_directory()) {
                        dirs.push_back(entry);
                    } else if (entry.is_regular_file()) {
                        auto ext = entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                        if (ext == ".ply" || ext == ".json" ||
                            entry.path().filename() == "cameras.bin" ||
                            entry.path().filename() == "transforms.json" ||
                            entry.path().filename() == "transforms_train.json") {
                            files.push_back(entry);
                        }
                    }
                }
            } catch (const std::filesystem::filesystem_error& e) {
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", e.what());
            }

            std::sort(dirs.begin(), dirs.end(), [](const auto& a, const auto& b) {
                return a.path().filename() < b.path().filename();
            });
            std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
                return a.path().filename() < b.path().filename();
            });

            for (const auto& dir : dirs) {
                std::string dirname = "[DIR] " + dir.path().filename().string();
                bool is_selected = (selected_file_ == dir.path().string());

                bool is_dataset = false;
                if (std::filesystem::exists(dir.path() / "sparse" / "0" / "cameras.bin") ||
                    std::filesystem::exists(dir.path() / "sparse" / "cameras.bin") ||
                    std::filesystem::exists(dir.path() / "transforms.json") ||
                    std::filesystem::exists(dir.path() / "transforms_train.json")) {
                    is_dataset = true;
                }

                if (is_dataset) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.5f, 0.9f, 1.0f));
                    dirname += " [Dataset]";
                }

                if (ImGui::Selectable(dirname.c_str(), is_selected,
                                      ImGuiSelectableFlags_AllowDoubleClick | ImGuiSelectableFlags_DontClosePopups)) {
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        current_path_ = dir.path().string();
                        selected_file_.clear();
                    } else {
                        selected_file_ = dir.path().string();
                    }
                }

                if (is_dataset) {
                    ImGui::PopStyleColor();
                }
            }

            for (const auto& file : files) {
                std::string filename = file.path().filename().string();
                bool is_selected = (selected_file_ == file.path().string());

                ImVec4 color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
                if (file.path().extension() == ".ply") {
                    color = ImVec4(0.3f, 0.8f, 0.3f, 1.0f);
                } else if (filename == "cameras.bin" || filename == "transforms.json" ||
                           filename == "transforms_train.json") {
                    color = ImVec4(0.3f, 0.5f, 0.9f, 1.0f);
                }

                ImGui::PushStyleColor(ImGuiCol_Text, color);
                if (ImGui::Selectable(filename.c_str(), is_selected, ImGuiSelectableFlags_DontClosePopups)) {
                    selected_file_ = file.path().string();
                }
                ImGui::PopStyleColor();
            }
        }
        ImGui::EndChild();

        if (!selected_file_.empty()) {
            ImGui::Text("Selected: %s", std::filesystem::path(selected_file_).filename().string().c_str());
        } else {
            ImGui::TextDisabled("No file selected");
        }

        ImGui::Separator();

        bool can_load = !selected_file_.empty();

        if (!can_load) {
            ImGui::BeginDisabled();
        }

        if (can_load) {
            std::filesystem::path selected_path(selected_file_);

            if (std::filesystem::is_directory(selected_path)) {
                bool is_colmap_dataset = false;
                bool is_transforms_dataset = false;

                if (std::filesystem::exists(selected_path / "sparse" / "0" / "cameras.bin") ||
                    std::filesystem::exists(selected_path / "sparse" / "cameras.bin")) {
                    is_colmap_dataset = true;
                }

                if (std::filesystem::exists(selected_path / "transforms.json") ||
                    std::filesystem::exists(selected_path / "transforms_train.json")) {
                    is_transforms_dataset = true;
                }

                if (is_colmap_dataset || is_transforms_dataset) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.8f, 1.0f));
                    if (ImGui::Button("Load Dataset", ImVec2(120, 0))) {
                        if (on_file_selected_) {
                            on_file_selected_(selected_path, true);
                            *p_open = false;
                        }
                    }
                    ImGui::PopStyleColor();

                    ImGui::SameLine();
                    ImGui::TextDisabled(is_colmap_dataset ? "(COLMAP)" : "(Transforms)");
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
                    if (ImGui::Button("Enter Directory", ImVec2(120, 0))) {
                        current_path_ = selected_path.string();
                        selected_file_.clear();
                    }
                    ImGui::PopStyleColor();

                    ImGui::SameLine();
                    ImGui::TextDisabled("(Not a dataset)");
                }
            } else {
                auto ext = selected_path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (ext == ".ply") {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                    if (ImGui::Button("Load PLY", ImVec2(120, 0))) {
                        if (on_file_selected_) {
                            on_file_selected_(selected_path, false);
                            *p_open = false;
                        }
                    }
                    ImGui::PopStyleColor();
                }
            }
        }

        if (!can_load) {
            ImGui::EndDisabled();
        }

        ImGui::SameLine();

        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            *p_open = false;
            selected_file_.clear();
        }

        ImGui::End();
        ImGui::PopStyleColor(2);
    }

    void FileBrowser::setOnFileSelected(std::function<void(const std::filesystem::path&, bool)> callback) {
        on_file_selected_ = callback;
    }

    void FileBrowser::setCurrentPath(const std::filesystem::path& path) {
        current_path_ = path.string();
    }

    void FileBrowser::setSelectedPath(const std::filesystem::path& path) {
        selected_file_ = path.string();
        if (std::filesystem::is_directory(path)) {
            current_path_ = path.string();
        } else {
            current_path_ = path.parent_path().string();
        }
    }
} // namespace gs::gui
