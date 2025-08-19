#pragma once

#include "input/input_handler.hpp"
#include "rendering/rendering_manager.hpp"
#include <string>
#include <string_view>

// Forward declaration for GLFW
struct GLFWwindow;

namespace gs {
    // Forward declarations
    class SceneManager;
    namespace visualizer {
        class RenderingManager;
    }
} // namespace gs

// Forward declaration for Viewport (check your viewport.hpp for actual namespace)
class Viewport;

namespace gs::gui {
    struct UIContext; // UIContext is in gs::gui namespace
}

namespace gs::visualizer {

    // Forward declarations
    class ToolContext;

    // C++23 concept defining what a tool must provide
    template <typename T>
    concept Tool = requires(T t, const ToolContext& ctx, const gs::gui::UIContext& ui_ctx, bool* p_open) {
        { t.getName() } -> std::convertible_to<std::string_view>;
        { t.getDescription() } -> std::convertible_to<std::string_view>;
        { t.isEnabled() } -> std::convertible_to<bool>;
        { t.setEnabled(bool{}) } -> std::same_as<void>;
        { t.initialize(ctx) } -> std::same_as<bool>;
        { t.shutdown() } -> std::same_as<void>;
        { t.update(ctx) } -> std::same_as<void>;
        { t.render(ctx) } -> std::same_as<void>;
        { t.renderUI(ui_ctx, p_open) } -> std::same_as<void>;
    };

    // Context passed to tools for accessing visualizer resources
    class ToolContext {
    public:
        virtual ~ToolContext() = default;

        // Access to visualizer components - make these const
        virtual RenderingManager* getRenderingManager() const = 0;
        virtual gs::SceneManager* getSceneManager() const = 0;
        virtual const ::Viewport& getViewport() const = 0;
        virtual ::GLFWwindow* getWindow() const = 0;

        // Tool-specific helpers
        virtual void requestRender() = 0;
        virtual void logMessage(const std::string& msg) = 0;
    };

    // Base class providing default implementations
    class ToolBase {
    public:
        virtual ~ToolBase() = default;

        virtual std::string_view getName() const = 0;
        virtual std::string_view getDescription() const = 0;

        bool isEnabled() const { return enabled_; }
        void setEnabled(bool enabled) {
            if (enabled_ != enabled) {
                enabled_ = enabled;
                onEnabledChanged(enabled);
            }
        }

        virtual bool initialize([[maybe_unused]] const ToolContext& ctx) { return true; }
        virtual void shutdown() {}
        virtual void update([[maybe_unused]] const ToolContext& ctx) {}
        virtual void render([[maybe_unused]] const ToolContext& ctx) {}
        virtual void renderUI(const gs::gui::UIContext& ui_ctx, bool* p_open) = 0;

    protected:
        virtual void onEnabledChanged([[maybe_unused]] bool enabled) {}

    private:
        bool enabled_ = false;
    };

} // namespace gs::visualizer