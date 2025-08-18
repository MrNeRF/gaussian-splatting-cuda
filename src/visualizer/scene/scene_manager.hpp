#pragma once

#include "core/events.hpp"
#include "core/parameters.hpp"
#include "rendering/rendering.hpp"
#include "scene/scene.hpp"
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>

namespace gs {

    // Forward declarations
    class TrainerManager;
    class SplatData;

    /**
     * @brief Manages scene lifecycle and coordinates between components
     *
     * The SceneManager is responsible for:
     * - Loading and managing scenes (PLY files, datasets)
     * - Coordinating between Scene, TrainerManager, and other components
     * - Managing scene state transitions
     * - Handling all scene-related events
     */
    class SceneManager {
    public:
        enum class SceneType {
            None,
            PLY,
            Dataset
        };

        struct SceneState {
            SceneType type = SceneType::None;
            std::filesystem::path source_path;
            size_t num_gaussians = 0;
            bool is_training = false;
            std::optional<int> training_iteration;
            size_t num_plys = 0; // For PLY mode
        };

        SceneManager();
        ~SceneManager();

        // Delete copy operations
        SceneManager(const SceneManager&) = delete;
        SceneManager& operator=(const SceneManager&) = delete;

        // Scene management
        void setScene(std::unique_ptr<Scene> scene);
        Scene* getScene() { return scene_.get(); }
        const Scene* getScene() const { return scene_.get(); }

        // Trainer management link
        void setTrainerManager(TrainerManager* trainer_manager) {
            trainer_manager_ = trainer_manager;
        }

        // High-level operations
        void loadPLY(const std::filesystem::path& path);
        void addPLY(const std::filesystem::path& path); // New method
        void loadDataset(const std::filesystem::path& path,
                         const param::TrainingParameters& params);
        void clearScene();

        // State queries
        SceneState getCurrentState() const;
        bool hasScene() const { return scene_ != nullptr && scene_->hasModel(); }

        // Rendering request
        gs::rendering::RenderingPipelineResult render(
            const gs::rendering::RenderingPipelineRequest& request);

        TrainerManager* getTrainerManager() { return trainer_manager_; }
        const TrainerManager* getTrainerManager() const { return trainer_manager_; }

    private:
        // Event handlers
        void setupEventHandlers();

        // Internal operations
        void loadPLYInternal(const std::filesystem::path& path);
        void addPLYInternal(const std::filesystem::path& path);
        void loadDatasetInternal(const std::filesystem::path& path,
                                 const param::TrainingParameters& params);
        void updateSceneState();

        // Member variables
        std::unique_ptr<Scene> scene_;
        TrainerManager* trainer_manager_ = nullptr;

        mutable std::mutex state_mutex_;
        SceneState current_state_;

        // Cached parameters for dataset loading
        std::optional<param::TrainingParameters> cached_params_;
    };

} // namespace gs