#pragma once

#include "core/events.hpp"
#include "tool_registry.hpp"
#include <memory>
#include <vector>

// Forward declaration for GLFW
struct GLFWwindow;

// Forward declaration for Viewport
class Viewport;

namespace gs {
    class SceneManager;

    namespace gui {
        struct UIContext;
    }

    namespace visualizer {
        class RenderingManager;
        class VisualizerImpl;

        class ToolManager : public ToolContext {
        public:
            explicit ToolManager(VisualizerImpl* visualizer);
            ~ToolManager();

            // Tool management
            void registerBuiltinTools();
            bool addTool(const std::string& tool_name);
            void removeTool(const std::string& tool_name);
            void removeAllTools();

            // Get active tools
            std::vector<ToolBase*> getActiveTools();
            ToolBase* getTool(const std::string& name);

            // Get tool registry for custom tool registration
            ToolRegistry& getRegistry() { return registry_; }

            // Lifecycle
            void initialize();
            void shutdown();
            void update();
            void render();
            void renderUI(const gs::gui::UIContext& ui_ctx);

            // ToolContext implementation - add const to match base class
            RenderingManager* getRenderingManager() const override;
            gs::SceneManager* getSceneManager() const override;
            const ::Viewport& getViewport() const override;
            ::GLFWwindow* getWindow() const override;
            void requestRender() override;
            void logMessage(const std::string& msg) override;

        private:
            void setupEventHandlers();

            VisualizerImpl* visualizer_;
            ToolRegistry registry_;
            std::vector<std::unique_ptr<ToolBase>> active_tools_;
            bool initialized_ = false;
        };

    } // namespace visualizer
} // namespace gs