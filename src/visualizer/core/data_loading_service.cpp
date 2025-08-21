#include "core/data_loading_service.hpp"
#include "scene/scene_manager.hpp"
#include <print>

namespace gs::visualizer {

    DataLoadingService::DataLoadingService(SceneManager* scene_manager)
        : scene_manager_(scene_manager) {
        setupEventHandlers();
    }

    DataLoadingService::~DataLoadingService() = default;

    void DataLoadingService::setupEventHandlers() {
        using namespace events;

        // Listen for file load commands
        cmd::LoadFile::when([this](const auto& cmd) {
            handleLoadFileCommand(cmd);
        });
    }

    void DataLoadingService::handleLoadFileCommand(const events::cmd::LoadFile& cmd) {
        if (cmd.is_dataset) {
            loadDataset(cmd.path);
        } else {
            // Check if we should add or replace
            if (scene_manager_->hasPLYFiles()) { // FIXED: Changed from isViewing()
                // In PLY viewing mode, add to existing
                scene_manager_->addPLY(cmd.path);
            } else {
                // Not in viewing mode - load as new scene
                scene_manager_->loadPLY(cmd.path);
            }
        }
    }

    std::expected<void, std::string> DataLoadingService::loadPLY(const std::filesystem::path& path) {
        try {
            std::println("Loading PLY file: {}", path.string());

            // Load through scene manager
            scene_manager_->loadPLY(path);

            // Emit success event
            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = std::format("Successfully loaded PLY: {}", path.filename().string()),
                .source = "DataLoadingService"}
                .emit();

            return {};
        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load PLY: {}", e.what());

            events::notify::Error{
                .message = error_msg,
                .details = std::format("Path: {}", path.string())}
                .emit();

            return std::unexpected(error_msg);
        }
    }

    void DataLoadingService::addPLYToScene(const std::filesystem::path& path) {
        try {
            std::println("Adding PLY to scene: {}", path.string());

            // Extract name from path
            std::string name = path.stem().string();

            // Add through scene manager
            scene_manager_->addPLY(path, name);

            // Log success
            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = std::format("Added PLY '{}' to scene", name),
                .source = "DataLoadingService"}
                .emit();

        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to add PLY: {}", e.what());

            events::notify::Error{
                .message = error_msg,
                .details = std::format("Path: {}", path.string())}
                .emit();
        }
    }

    std::expected<void, std::string> DataLoadingService::loadDataset(const std::filesystem::path& path) {
        try {
            std::println("Loading dataset from: {}", path.string());

            // Validate parameters
            if (params_.dataset.data_path.empty() && path.empty()) {
                throw std::runtime_error("No dataset path specified");
            }

            // Load through scene manager
            scene_manager_->loadDataset(path, params_);

            // Emit success event
            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = std::format("Successfully loaded dataset: {}", path.filename().string()),
                .source = "DataLoadingService"}
                .emit();

            return {};
        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load dataset: {}", e.what());

            events::notify::Error{
                .message = error_msg,
                .details = std::format("Path: {}", path.string())}
                .emit();

            return std::unexpected(error_msg);
        }
    }

    void DataLoadingService::clearScene() {
        try {
            scene_manager_->clear();

            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = "Scene cleared",
                .source = "DataLoadingService"}
                .emit();
        } catch (const std::exception& e) {
            events::notify::Error{
                .message = "Failed to clear scene",
                .details = e.what()}
                .emit();
        }
    }

} // namespace gs::visualizer