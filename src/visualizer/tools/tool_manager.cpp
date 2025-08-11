#include "tools/tool_manager.hpp"
#include "core/events.hpp"
#include "tools/crop_box_tool.hpp"
#include "visualizer_impl.hpp"
#include "world_transform_tool.hpp"

#include <algorithm>
#include <print>

namespace gs::visualizer {

    ToolManager::ToolManager(VisualizerImpl* visualizer)
        : visualizer_(visualizer) {
        setupEventHandlers();
    }

    ToolManager::~ToolManager() {
        shutdown();
    }

    void ToolManager::registerBuiltinTools() {
        // Register all built-in tools
        registry_.registerTool<CropBoxTool>();
        registry_.registerTool<WorldTransformTool>();

        // Future tools would be registered here:
    }

    bool ToolManager::addTool(const std::string& tool_name) {
        // Check if tool is already active
        auto existing = std::find_if(active_tools_.begin(), active_tools_.end(),
                                     [&](const auto& tool) { return tool->getName() == tool_name; });

        if (existing != active_tools_.end()) {
            std::println("Tool '{}' is already active", tool_name);
            return false;
        }

        // Create new tool instance
        auto tool = registry_.createTool(tool_name);
        if (!tool) {
            std::println("Failed to create tool '{}'", tool_name);
            return false;
        }

        // Initialize if manager is already initialized
        if (initialized_) {
            if (!tool->initialize(*this)) {
                std::println("Failed to initialize tool '{}'", tool_name);
                return false;
            }
        }

        active_tools_.push_back(std::move(tool));

        events::notify::Log{
            .level = events::notify::Log::Level::Info,
            .message = std::format("Added tool: {}", tool_name),
            .source = "ToolManager"}
            .emit();

        return true;
    }

    void ToolManager::removeTool(const std::string& tool_name) {
        auto it = std::find_if(active_tools_.begin(), active_tools_.end(),
                               [&](const auto& tool) { return tool->getName() == tool_name; });

        if (it != active_tools_.end()) {
            (*it)->shutdown();
            active_tools_.erase(it);

            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = std::format("Removed tool: {}", tool_name),
                .source = "ToolManager"}
                .emit();
        }
    }

    void ToolManager::removeAllTools() {
        for (auto& tool : active_tools_) {
            tool->shutdown();
        }
        active_tools_.clear();
    }

    std::vector<ToolBase*> ToolManager::getActiveTools() {
        std::vector<ToolBase*> tools;
        for (auto& tool : active_tools_) {
            tools.push_back(tool.get());
        }
        return tools;
    }

    ToolBase* ToolManager::getTool(const std::string& name) {
        auto it = std::find_if(active_tools_.begin(), active_tools_.end(),
                               [&](const auto& tool) { return tool->getName() == name; });

        if (it != active_tools_.end()) {
            return it->get();
        }
        return nullptr;
    }

    void ToolManager::initialize() {
        for (auto& tool : active_tools_) {
            if (!tool->initialize(*this)) {
                std::println("Warning: Failed to initialize tool '{}'", tool->getName());
            }
        }
        initialized_ = true;
    }

    void ToolManager::shutdown() {
        for (auto& tool : active_tools_) {
            tool->shutdown();
        }
        initialized_ = false;
    }

    void ToolManager::update() {
        for (auto& tool : active_tools_) {
            if (tool->isEnabled()) {
                tool->update(*this);
            }
        }
    }

    void ToolManager::render() {
        for (auto& tool : active_tools_) {
            if (tool->isEnabled()) {
                tool->render(*this);
            }
        }
    }

    void ToolManager::renderUI(const gui::UIContext& ui_ctx) {
        for (auto& tool : active_tools_) {
            bool dummy_open = true;
            tool->renderUI(ui_ctx, &dummy_open);
        }
    }

    // ToolContext implementation
    RenderingManager* ToolManager::getRenderingManager() {
        return visualizer_->rendering_manager_.get();
    }

    gs::SceneManager* ToolManager::getSceneManager() {
        return visualizer_->scene_manager_.get();
    }

    const ::Viewport& ToolManager::getViewport() const {
        return visualizer_->viewport_;
    }

    ::GLFWwindow* ToolManager::getWindow() const {
        return visualizer_->getWindow();
    }

    void ToolManager::requestRender() {
        // Could implement frame request logic here if needed
    }

    void ToolManager::logMessage(const std::string& msg) {
        if (visualizer_->gui_manager_) {
            visualizer_->gui_manager_->addConsoleLog("%s", msg.c_str());
        }
    }

    void ToolManager::setupEventHandlers() {
        using namespace events;

        // Listen for tool enable/disable events
        tools::ToolEnabled::when([this](const auto& e) {
            if (auto* tool = getTool(e.tool_name)) {
                tool->setEnabled(true);
            }
        });

        tools::ToolDisabled::when([this](const auto& e) {
            if (auto* tool = getTool(e.tool_name)) {
                tool->setEnabled(false);
            }
        });
    }

} // namespace gs::visualizer