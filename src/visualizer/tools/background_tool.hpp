#pragma once

#include "tools/tool_base.hpp"
#include <glm/glm.hpp>

namespace gs::visualizer {

    class BackgroundTool : public ToolBase {
    public:
        BackgroundTool();
        ~BackgroundTool() override;

        std::string_view getName() const override { return "Background"; }
        std::string_view getDescription() const override {
            return "Control background color";
        }

        bool initialize(const ToolContext& ctx) override;
        void shutdown() override;
        void update(const ToolContext& ctx) override;
        void render(const ToolContext& ctx) override;
        void renderUI(const gs::gui::UIContext& ui_ctx, bool* p_open) override;

        // Get current background color
        glm::vec3 getBackgroundColor() const { return background_color_; }

    protected:
        void onEnabledChanged(bool enabled) override;

    private:
        glm::vec3 background_color_;
        float color_array_[3];

        // Helper to convert color_array_ to vec3
        glm::vec3 getColorArrayAsVec3() const;
    };

} // namespace gs::visualizer