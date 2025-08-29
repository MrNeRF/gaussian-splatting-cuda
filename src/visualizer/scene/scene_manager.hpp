/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "core/parameters.hpp"
#include "scene/scene.hpp"
#include <filesystem>
#include <mutex>

namespace gs {

    // Forward declarations
    class Trainer;
    class TrainerManager;
    class SplatData;

    class SceneManager {
    public:
        // Content type - what's loaded, not execution state
        enum class ContentType {
            Empty,
            PLYFiles,
            Dataset
        };

        SceneManager();
        ~SceneManager();

        // Delete copy operations
        SceneManager(const SceneManager&) = delete;
        SceneManager& operator=(const SceneManager&) = delete;

        // Content queries - direct, no events
        ContentType getContentType() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return content_type_;
        }

        bool isEmpty() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return content_type_ == ContentType::Empty;
        }

        bool hasPLYFiles() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return content_type_ == ContentType::PLYFiles;
        }

        bool hasDataset() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return content_type_ == ContentType::Dataset;
        }

        // Path accessors
        std::vector<std::filesystem::path> getPLYPaths() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return ply_paths_;
        }

        std::filesystem::path getDatasetPath() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return dataset_path_;
        }

        // Scene access
        Scene& getScene() { return scene_; }
        const Scene& getScene() const { return scene_; }

        // Trainer manager link
        void setTrainerManager(TrainerManager* tm) { trainer_manager_ = tm; }
        TrainerManager* getTrainerManager() { return trainer_manager_; }
        const TrainerManager* getTrainerManager() const { return trainer_manager_; }

        void changeContentType(const ContentType& type);

        // Operations
        void loadPLY(const std::filesystem::path& path);
        void addPLY(const std::filesystem::path& path, const std::string& name = "", bool is_visible = true);
        void removePLY(const std::string& name);
        void setPLYVisibility(const std::string& name, bool visible);

        void loadDataset(const std::filesystem::path& path,
                         const param::TrainingParameters& params);
        void clear();

        // For rendering - gets appropriate model
        const SplatData* getModelForRendering() const;

        // Direct info queries
        struct SceneInfo {
            bool has_model = false;
            size_t num_gaussians = 0;
            size_t num_nodes = 0;
            std::string source_type;
            std::filesystem::path source_path;
        };

        SceneInfo getSceneInfo() const;

    private:
        void setupEventHandlers();
        void emitSceneChanged();

        Scene scene_;
        mutable std::mutex state_mutex_;

        ContentType content_type_ = ContentType::Empty;
        std::vector<std::filesystem::path> ply_paths_;
        std::filesystem::path dataset_path_;

        // Training support
        TrainerManager* trainer_manager_ = nullptr;

        // Cache for parameters
        std::optional<param::TrainingParameters> cached_params_;
    };

} // namespace gs