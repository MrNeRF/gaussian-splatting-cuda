/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/data_loading_service.hpp"
#include "core/logger.hpp"
#include "scene/scene_manager.hpp"
#include <stdexcept>

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
            if (scene_manager_->hasPLYFiles()) {
                // In PLY viewing mode, add to existing
                scene_manager_->addPLY(cmd.path);
            } else {
                // Not in viewing mode - load as new scene
                scene_manager_->loadPLY(cmd.path);
            }
        }
    }

    std::expected<void, std::string> DataLoadingService::loadPLY(const std::filesystem::path& path) {
        LOG_TIMER("LoadPLY");

        try {
            LOG_INFO("Loading PLY file: {}", path.string());

            // Load through scene manager
            scene_manager_->loadPLY(path);

            LOG_INFO("Successfully loaded PLY: {} (from: {})",
                     path.filename().string(),
                     path.parent_path().string());

            return {};
        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load PLY: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, path.string());
            throw std::runtime_error(error_msg);
        }
    }

    void DataLoadingService::addPLYToScene(const std::filesystem::path& path) {
        LOG_TIMER_TRACE("AddPLYToScene");

        try {
            LOG_DEBUG("Adding PLY to scene: {}", path.string());

            // Extract name from path
            std::string name = path.stem().string();
            LOG_TRACE("Extracted PLY name: {}", name);

            // Add through scene manager
            scene_manager_->addPLY(path, name);

            LOG_INFO("Added PLY '{}' to scene", name);

        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to add PLY: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, path.string());
            throw std::runtime_error(error_msg);
        }
    }

    std::expected<void, std::string> DataLoadingService::loadDataset(const std::filesystem::path& path) {
        LOG_TIMER("LoadDataset");

        try {
            LOG_INFO("Loading dataset from: {}", path.string());

            // Validate parameters
            if (params_.dataset.data_path.empty() && path.empty()) {
                LOG_ERROR("No dataset path specified");
                throw std::runtime_error("No dataset path specified");
            }

            // Load through scene manager
            LOG_DEBUG("Passing dataset to scene manager with parameters");
            scene_manager_->loadDataset(path, params_);

            return {};
        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load dataset: {}", e.what());
            LOG_ERROR("{} (Path: {})", error_msg, path.string());
            throw std::runtime_error(error_msg);
        }
    }

    void DataLoadingService::clearScene() {
        try {
            LOG_DEBUG("Clearing scene");
            scene_manager_->clear();
            LOG_INFO("Scene cleared");
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to clear scene: {}", e.what());
            throw std::runtime_error(std::format("Failed to clear scene: {}", e.what()));
        }
    }

} // namespace gs::visualizer