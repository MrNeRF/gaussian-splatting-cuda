#pragma once

#include "input/input_handler.hpp"
#include "input/input_priority.hpp"
#include <memory>
#include <string>
#include <string_view>
#include <vector>

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

        // Access to visualizer components
        virtual RenderingManager* getRenderingManager() = 0;
        virtual gs::SceneManager* getSceneManager() = 0;
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

        virtual bool initialize(const ToolContext& ctx) { return true; }
        virtual void shutdown() {}
        virtual void update(const ToolContext& ctx) {}
        virtual void render(const ToolContext& ctx) {}
        virtual void renderUI(const gs::gui::UIContext& ui_ctx, bool* p_open) = 0;

        // Input handling - tools can override these to register handlers
        virtual void registerInputHandlers(InputHandler& handler) {}
        virtual void unregisterInputHandlers(InputHandler& handler) {
            for (auto id : handler_ids_) {
                handler.removeHandler(id);
            }
            handler_ids_.clear();
        }

    protected:
        virtual void onEnabledChanged(bool enabled) {}

        // Store handler IDs for cleanup
        std::vector<InputHandler::HandlerId> handler_ids_;

    private:
        bool enabled_ = false;
    };

} // namespace gs::visualizer
