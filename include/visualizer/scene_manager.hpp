#pragma once

#include "core/parameters.hpp"
#include "visualizer/event_bus.hpp"
#include "visualizer/events.hpp"
#include "visualizer/rendering_pipeline.hpp"
#include "visualizer/scene.hpp"
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
        };

        explicit SceneManager(std::shared_ptr<EventBus> event_bus);
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
        void loadDataset(const std::filesystem::path& path,
                         const param::TrainingParameters& params);
        void clearScene();

        // State queries
        SceneState getCurrentState() const;
        bool hasScene() const { return scene_ != nullptr && scene_->hasModel(); }

        // Rendering request
        RenderingPipeline::RenderResult render(
            const RenderingPipeline::RenderRequest& request);

    private:
        // Event handlers
        void setupEventHandlers();

        // Command handlers
        void handleLoadFileCommand(const LoadFileCommand& cmd);
        void handleClearSceneCommand(const ClearSceneCommand& cmd);
        void handleRenderRequestCommand(const RenderRequestCommand& cmd);

        // Query handlers
        void handleSceneStateQuery(const QuerySceneStateRequest& request);
        void handleRenderCapabilitiesQuery(const QueryRenderCapabilitiesRequest& request);

        // Training event handlers
        void handleTrainingStarted(const TrainingStartedEvent& event);
        void handleTrainingCompleted(const TrainingCompletedEvent& event);
        void handleModelUpdated(const ModelUpdatedEvent& event);

        // Internal operations
        void loadPLYInternal(const std::filesystem::path& path);
        void loadDatasetInternal(const std::filesystem::path& path,
                                 const param::TrainingParameters& params);
        void updateSceneState();
        void publishSceneStateChanged(const SceneState& old_state,
                                      const SceneState& new_state);

        // Member variables
        std::shared_ptr<EventBus> event_bus_;
        std::vector<size_t> event_handler_ids_;

        std::unique_ptr<Scene> scene_;
        TrainerManager* trainer_manager_ = nullptr;

        mutable std::mutex state_mutex_;
        SceneState current_state_;

        // Cached parameters for dataset loading
        std::optional<param::TrainingParameters> cached_params_;
    };

} // namespace gs
