#pragma once

#include "core/events.hpp"
#include "core/trainer.hpp"
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>

namespace gs {

    // Forward declarations
    namespace visualizer {
        class VisualizerImpl;
    }

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

        TrainerManager();
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
        bool hasTrainer() const;

        // Link to viewer for notifications
        void setViewer(visualizer::VisualizerImpl* viewer) { viewer_ = viewer; }

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

        // call the trainer getCamById
        std::shared_ptr<const Camera> getCamById(int camId) const;

        std::vector<std::shared_ptr<const Camera>> getCamList() const;

    private:
        // Training thread function
        void trainingThreadFunc(std::stop_token stop_token);

        // State management
        void setState(State new_state);
        void handleTrainingComplete(bool success, const std::string& error = "");

        // Member variables
        std::unique_ptr<Trainer> trainer_;
        std::unique_ptr<std::jthread> training_thread_;
        visualizer::VisualizerImpl* viewer_ = nullptr;

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