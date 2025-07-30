#pragma once

#include "scene/scene_manager.hpp"
#include <functional>
#include <string>
#include <unordered_map>

namespace gs {

    class CommandProcessor {
    public:
        using CommandHandler = std::function<std::string()>;

        explicit CommandProcessor(SceneManager* scene_manager);

        // Process a command and return the result
        std::string processCommand(const std::string& command);

        // Register custom commands
        void registerCommand(const std::string& name, CommandHandler handler);

    private:
        // Built-in command handlers
        std::string handleHelp();
        std::string handleStatus();
        std::string handleModelInfo();
        std::string handleGpuInfo();
        std::string handleTensorInfo(const std::string& tensor_name);

        SceneManager* scene_manager_;
        std::unordered_map<std::string, CommandHandler> commands_;

        void registerBuiltinCommands();
    };

} // namespace gs
