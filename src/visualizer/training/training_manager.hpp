#pragma once

#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/parameters.hpp"
#include "core/trainer.hpp"
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>

namespace gs {

    // Forward declarations
    class GSViewer;

    /**
     * @brief Manages training lifecycle and thread coordination
     *
     * This class encapsulates all training-related functionality including:
     * - Training thread management
     * - State control (start/pause/resume/stop)
     * - Progress tracking and reporting
     * - Communication between trainer and viewer
     */
    class TrainerManager {
    public:
        enum class State {
            Idle,      // No trainer loaded
            Ready,     // Trainer loaded, ready to start
            Running,   // Training in progress
            Paused,    // Training paused
            Stopping,  // Stop requested, waiting for thread
            Completed, // Training finished successfully
            Error      // Training encountered an error
        };

        TrainerManager() = default;
        ~TrainerManager();

        // Delete copy operations
        TrainerManager(const TrainerManager&) = delete;
        TrainerManager& operator=(const TrainerManager&) = delete;

        // Allow move operations
        TrainerManager(TrainerManager&&) = default;
        TrainerManager& operator=(TrainerManager&&) = default;

        // Setup and teardown
        void setTrainer(std::unique_ptr<Trainer> trainer);
        void clearTrainer();
        bool hasTrainer() const { return trainer_ != nullptr; }

        // Link to viewer for notifications
        void setViewer(GSViewer* viewer) { viewer_ = viewer; }

        // Set event bus for publishing training events
        void setEventBus(std::shared_ptr<EventBus> event_bus);

        // Training control
        bool startTraining();
        void pauseTraining();
        void resumeTraining();
        void stopTraining();
        void requestSaveCheckpoint();

        // State queries
        State getState() const { return state_.load(); }
        bool isRunning() const { return state_ == State::Running; }
        bool isPaused() const { return state_ == State::Paused; }
        bool isTrainingActive() const {
            auto s = state_.load();
            return s == State::Running || s == State::Paused;
        }
        bool canStart() const { return state_ == State::Ready; }
        bool canPause() const { return state_ == State::Running; }
        bool canResume() const { return state_ == State::Paused; }
        bool canStop() const { return isTrainingActive(); }

        // Progress information
        int getCurrentIteration() const;
        float getCurrentLoss() const;
        int getTotalIterations() const;

        // Access to trainer (for rendering, etc.)
        Trainer* getTrainer() { return trainer_.get(); }
        const Trainer* getTrainer() const { return trainer_.get(); }

        // Wait for training to complete (blocking)
        void waitForCompletion();

        // Get last error message
        const std::string& getLastError() const { return last_error_; }

    private:
        // Training thread function
        void trainingThreadFunc(std::stop_token stop_token);

        // State management
        void setState(State new_state);
        void handleTrainingComplete(bool success, const std::string& error = "");

        // Event handlers
        void handleStateQueryRequest(const QueryTrainerStateRequest& request);
        void publishStateChange(State old_state, State new_state, const std::string& reason);

        // Publish training events
        void publishTrainingStarted(int total_iterations);
        void publishTrainingProgress(int iteration, float loss, int num_gaussians, bool is_refining);
        void publishTrainingPaused(int iteration);
        void publishTrainingResumed(int iteration);
        void publishTrainingCompleted(int iteration, float loss, bool success, const std::string& error = "");
        void publishTrainingStopped(int iteration, bool user_requested);

        // Member variables
        std::unique_ptr<Trainer> trainer_;
        std::unique_ptr<std::jthread> training_thread_;
        GSViewer* viewer_ = nullptr;

        // Event bus for publishing events
        std::shared_ptr<EventBus> event_bus_;

        // State tracking
        std::atomic<State> state_{State::Idle};
        std::string last_error_;
        mutable std::mutex state_mutex_;

        // Synchronization
        std::condition_variable completion_cv_;
        std::mutex completion_mutex_;
        bool training_complete_ = false;
    };

} // namespace gs