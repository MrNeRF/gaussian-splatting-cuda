#include "visualizer/training_manager.hpp"
#include "visualizer/detail.hpp"
#include <print>

namespace gs {

    TrainerManager::~TrainerManager() {
        // Ensure training is stopped before destruction
        if (training_thread_ && training_thread_->joinable()) {
            std::println("TrainerManager: Stopping training thread...");
            stopTraining();
            waitForCompletion();
        }
    }

    void TrainerManager::setTrainer(std::unique_ptr<Trainer> trainer) {
        // Clear any existing trainer first
        clearTrainer();

        if (trainer) {
            trainer_ = std::move(trainer);
            setState(State::Ready);

            // Just set event bus
            if (event_bus_) {
                trainer_->setEventBus(event_bus_);
            }
        }
    }

    void TrainerManager::clearTrainer() {
        // Stop any ongoing training
        if (isTrainingActive()) {
            stopTraining();
            waitForCompletion();
        }

        // Clear the trainer
        trainer_.reset();
        last_error_.clear();
        setState(State::Idle);
    }

    bool TrainerManager::startTraining() {
        if (!canStart()) {
            std::println("TrainerManager: Cannot start training in current state: {}",
                         static_cast<int>(state_.load()));
            return false;
        }

        if (!trainer_) {
            std::println("TrainerManager: No trainer available");
            return false;
        }

        // Reset completion state
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            training_complete_ = false;
        }

        // Publish training started event
        if (event_bus_) {
            publishTrainingStarted(getTotalIterations());
        }

        // Start training thread
        setState(State::Running);
        training_thread_ = std::make_unique<std::jthread>(
            [this](std::stop_token stop_token) {
                trainingThreadFunc(stop_token);
            });

        std::println("TrainerManager: Training started");
        return true;
    }

    void TrainerManager::pauseTraining() {
        if (!canPause()) {
            return;
        }

        if (trainer_) {
            trainer_->request_pause();
            setState(State::Paused);

            if (event_bus_) {
                publishTrainingPaused(getCurrentIteration());
            }

            std::println("TrainerManager: Training paused");
        }
    }

    void TrainerManager::resumeTraining() {
        if (!canResume()) {
            return;
        }

        if (trainer_) {
            trainer_->request_resume();
            setState(State::Running);

            if (event_bus_) {
                publishTrainingResumed(getCurrentIteration());
            }

            std::println("TrainerManager: Training resumed");
        }
    }

    void TrainerManager::stopTraining() {
        if (!isTrainingActive()) {
            return;
        }

        setState(State::Stopping);

        if (trainer_) {
            trainer_->request_stop();
        }

        if (training_thread_ && training_thread_->joinable()) {
            std::println("TrainerManager: Requesting training thread to stop...");
            training_thread_->request_stop();
        }

        if (event_bus_) {
            publishTrainingStopped(getCurrentIteration(), true);
        }
    }

    void TrainerManager::requestSaveCheckpoint() {
        if (trainer_ && isTrainingActive()) {
            trainer_->request_save();
            std::println("TrainerManager: Checkpoint save requested");

            if (event_bus_) {
                // This will be handled by the trainer when it actually saves
                // For now, just log it
                event_bus_->publish(LogMessageEvent{
                    LogMessageEvent::Level::Info,
                    "Checkpoint save requested",
                    "TrainerManager"});
            }
        }
    }

    void TrainerManager::waitForCompletion() {
        if (!training_thread_ || !training_thread_->joinable()) {
            return;
        }

        std::unique_lock<std::mutex> lock(completion_mutex_);
        completion_cv_.wait(lock, [this] { return training_complete_; });

        training_thread_->join();
        training_thread_.reset();
    }

    int TrainerManager::getCurrentIteration() const {
        return trainer_ ? trainer_->get_current_iteration() : 0;
    }

    float TrainerManager::getCurrentLoss() const {
        return trainer_ ? trainer_->get_current_loss() : 0.0f;
    }

    int TrainerManager::getTotalIterations() const {
        if (!trainer_)
            return 0;

        // This is a bit of a hack - we'd need to expose this from Trainer
        // For now, return a default value
        return 30000; // Default iterations
    }

    void TrainerManager::trainingThreadFunc(std::stop_token stop_token) {
        std::println("TrainerManager: Training thread started");

        try {
            // Set up progress callback for the trainer
            if (trainer_ && event_bus_) {
                // Note: The trainer now has the event bus and will publish progress events
            }

            auto train_result = trainer_->train(stop_token);

            if (!train_result) {
                handleTrainingComplete(false, train_result.error());
            } else {
                handleTrainingComplete(true);
            }
        } catch (const std::exception& e) {
            handleTrainingComplete(false, std::format("Exception in training: {}", e.what()));
        } catch (...) {
            handleTrainingComplete(false, "Unknown exception in training");
        }

        std::println("TrainerManager: Training thread finished");
    }

    void TrainerManager::setState(State new_state) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_ = new_state;
    }

    void TrainerManager::handleTrainingComplete(bool success, const std::string& error) {
        if (!error.empty()) {
            last_error_ = error;
            std::println(stderr, "TrainerManager: Training error: {}", error);
        }

        setState(success ? State::Completed : State::Error);

        if (event_bus_) {
            publishTrainingCompleted(
                getCurrentIteration(),
                getCurrentLoss(),
                success,
                error);
        }

        // Notify completion
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            training_complete_ = true;
        }
        completion_cv_.notify_all();
    }

    // Event publishing methods
    void TrainerManager::publishTrainingStarted(int total_iterations) {
        if (event_bus_) {
            event_bus_->publish(TrainingStartedEvent{total_iterations});

            // Signal trainer to start
            event_bus_->publish(TrainingReadyToStartEvent{});
        }
    }

    void TrainerManager::publishTrainingProgress(int iteration, float loss, int num_gaussians, bool is_refining) {
        if (event_bus_) {
            event_bus_->publish(TrainingProgressEvent{
                iteration,
                getTotalIterations(),
                loss,
                num_gaussians,
                is_refining});
        }
    }

    void TrainerManager::publishTrainingPaused(int iteration) {
        if (event_bus_) {
            event_bus_->publish(TrainingPausedEvent{iteration});
        }
    }

    void TrainerManager::publishTrainingResumed(int iteration) {
        if (event_bus_) {
            event_bus_->publish(TrainingResumedEvent{iteration});
        }
    }

    void TrainerManager::publishTrainingCompleted(int iteration, float loss, bool success, const std::string& error) {
        if (event_bus_) {
            event_bus_->publish(TrainingCompletedEvent{
                iteration,
                loss,
                success,
                error.empty() ? std::nullopt : std::optional<std::string>(error)});
        }
    }

    void TrainerManager::publishTrainingStopped(int iteration, bool user_requested) {
        if (event_bus_) {
            event_bus_->publish(TrainingStoppedEvent{iteration, user_requested});
        }
    }

} // namespace gs