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

        if (ImGui::ColorEdit3("Background Color", color_array_)) {
            background_color_ = getColorArrayAsVec3();

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
        if (enabled) {
            // Emit current color when enabled
            events::ui::RenderSettingsChanged{
                .fov = std::nullopt,
                .scaling_modifier = std::nullopt,
                .antialiasing = std::nullopt,
                .background_color = background_color_}
                .emit();
        }
    }

    glm::vec3 BackgroundTool::getColorArrayAsVec3() const {
        return glm::vec3(color_array_[0], color_array_[1], color_array_[2]);
    }

} // namespace gs::visualizer