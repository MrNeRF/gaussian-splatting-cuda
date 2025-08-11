#include "tools/background_tool.hpp"
#include "core/events.hpp"
#include <imgui.h>

namespace gs::visualizer {

    BackgroundTool::BackgroundTool()
        : background_color_(0.0f, 0.0f, 0.0f) {
        color_array_[0] = background_color_.r;
        color_array_[1] = background_color_.g;
        color_array_[2] = background_color_.b;
    }

    BackgroundTool::~BackgroundTool() = default;

    bool BackgroundTool::initialize([[maybe_unused]] const ToolContext& ctx) {
        return true;
    }

    void BackgroundTool::shutdown() {
        // Nothing to cleanup
    }

    void BackgroundTool::update([[maybe_unused]] const ToolContext& ctx) {
        // Nothing to update per frame
    }

    void BackgroundTool::render([[maybe_unused]] const ToolContext& ctx) {
        // No rendering needed
    }

    void BackgroundTool::renderUI([[maybe_unused]] const gs::gui::UIContext& ui_ctx, [[maybe_unused]] bool* p_open) {
        if (!isEnabled()) {
            return;
        }

        if (!ImGui::CollapsingHeader("Background")) {
            return;
        }

        bool color_changed = false;

        // Color picker without label
        if (ImGui::ColorEdit3("##BackgroundColor", color_array_)) {
            background_color_ = getColorArrayAsVec3();
            color_changed = true;
        }

        // Reset button
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            // Reset to black
            background_color_ = glm::vec3(0.0f, 0.0f, 0.0f);
            color_array_[0] = 0.0f;
            color_array_[1] = 0.0f;
            color_array_[2] = 0.0f;
            color_changed = true;
        }

        if (color_changed) {
            // Emit event to update rendering
            events::ui::RenderSettingsChanged{
                .fov = std::nullopt,
                .scaling_modifier = std::nullopt,
                .antialiasing = std::nullopt,
                .background_color = background_color_}
                .emit();
        }
    }

    void BackgroundTool::onEnabledChanged(bool enabled) {
        // Don't do anything when enabling/disabling
        // The color should persist regardless of tool state
        (void)enabled;
    }

    glm::vec3 BackgroundTool::getColorArrayAsVec3() const {
        return glm::vec3(color_array_[0], color_array_[1], color_array_[2]);
    }

} // namespace gs::visualizer