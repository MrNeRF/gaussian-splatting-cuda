// scene_manager.hpp
#pragma once

#include "core/events.hpp"
#include "core/parameters.hpp"
#include "scene/scene.hpp"
#include <filesystem>
#include <memory>
#include <mutex>
#include <variant>

namespace gs {

    // Forward declarations
    class Trainer;
    class TrainerManager;
    class SplatData;

    class SceneManager {
    public:
        // Clear state representation
        struct EmptyState {};

        struct ViewingState {
            std::vector<std::filesystem::path> ply_paths;
        };

        struct TrainingState {
            std::filesystem::path dataset_path;
            bool is_running = false;
            int current_iteration = 0;
        };

        using State = std::variant<EmptyState, ViewingState, TrainingState>;

        SceneManager();
        ~SceneManager();

        // Delete copy operations
        SceneManager(const SceneManager&) = delete;
        SceneManager& operator=(const SceneManager&) = delete;

        // State queries - direct, no events
        const State& getState() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return state_;
        }

        bool isEmpty() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return std::holds_alternative<EmptyState>(state_);
        }

        bool isViewing() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return std::holds_alternative<ViewingState>(state_);
        }

        bool isTraining() const {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return std::holds_alternative<TrainingState>(state_);
        }

        // Scene access
        Scene& getScene() { return scene_; }
        const Scene& getScene() const { return scene_; }

        // Trainer manager link
        void setTrainerManager(TrainerManager* tm) { trainer_manager_ = tm; }
        TrainerManager* getTrainerManager() { return trainer_manager_; }

        // Operations
        void loadPLY(const std::filesystem::path& path);
        void addPLY(const std::filesystem::path& path, const std::string& name = "");
        void removePLY(const std::string& name);
        void setPLYVisibility(const std::string& name, bool visible);

        void loadDataset(const std::filesystem::path& path,
                         const param::TrainingParameters& params);
        void clear();

        // For rendering - gets appropriate model
        const SplatData* getModelForRendering() const;

        // Direct info queries (replace events)
        struct SceneInfo {
            bool has_model = false;
            size_t num_gaussians = 0;
            size_t num_nodes = 0;
            std::string source_type;
            std::filesystem::path source_path;
        };

        SceneInfo getSceneInfo() const;

    private:
        void transitionTo(State new_state);
        void setupEventHandlers();
        void updateTrainingModel();
        void emitSceneChanged();

        Scene scene_;
        mutable std::mutex state_mutex_;
        State state_ = EmptyState{};

        // Training support
        TrainerManager* trainer_manager_ = nullptr;
        mutable std::unique_ptr<SplatData> training_model_snapshot_;
        mutable bool training_snapshot_valid_ = false;

        // Cache for parameters
        std::optional<param::TrainingParameters> cached_params_;
    };

} // namespace gs
