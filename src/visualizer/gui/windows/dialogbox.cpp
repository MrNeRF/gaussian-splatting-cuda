#include "gui/windows/dialogbox.hpp"
#include <imgui.h>
#include "gui/ui_widgets.hpp"
#include "visualizer/visualizer_impl.hpp" // <-- Add this include
#include <core/logger.hpp>

namespace gs::gui {

    SaveProjectDialogBox::SaveProjectDialogBox() {
        // current_path_ = std::filesystem::current_path().string();
    }

    void SaveProjectDialogBox::render(bool* p_open) {
        // Add NoDocking flag
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.15f, 0.9f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

        if (ImGui::Begin("Project was modified", p_open, window_flags)) {
            ImGui::Text("");
            ImGui::Text("Save Changes ?");

            ImGui::Text("");

            if (ImGui::BeginTable("button_table", 3)) {

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                if (ImGui::Button("Yes", ImVec2(80, 0))) {
                    if (result_) {
                        result_(true);
                    }
                    *p_open = false;
                }
                ImGui::TableNextColumn();
                if (ImGui::Button("No", ImVec2(80, 0))) {
                    if (result_) {
                        result_(false);
                    }
                    *p_open = false;
                }
                ImGui::TableNextColumn();
                if (ImGui::Button("Cancel", ImVec2(80, 0))) {
                    *p_open = false;
                }

                ImGui::EndTable();
            }

            ImGui::Text("");

        }
        ImGui::End();
        ImGui::PopStyleColor(4);
    }

    void SaveProjectDialogBox::setOnDialogClose(std::function<void(bool)> callback) {
        result_ = callback;
    }
} // namespace gs::gui
