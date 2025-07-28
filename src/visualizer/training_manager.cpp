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

            // Publish state change event
            publishStateChange(State::Idle, State::Ready, "Trainer loaded");
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

        // Publish state change
        auto old_state = state_.load();
        if (old_state != State::Idle) {
            publishStateChange(old_state, State::Idle, "Trainer cleared");
        }
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

        // Publish state change
        publishStateChange(State::Ready, State::Running, "Training started");
        setState(State::Running);

        // Publish training started event
        if (event_bus_) {
            publishTrainingStarted(getTotalIterations());
        }

        // Start training thread
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
            publishStateChange(State::Running, State::Paused, "User requested pause");
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
            publishStateChange(State::Paused, State::Running, "User requested resume");
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

        auto old_state = state_.load();
        publishStateChange(old_state, State::Stopping, "User requested stop");
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
        return trainer_->getParams().optimization.iterations;
    }

    void TrainerManager::trainingThreadFunc(std::stop_token stop_token) {
        std::println("TrainerManager: Training thread started");

        try {
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

        auto old_state = state_.load();
        auto new_state = success ? State::Completed : State::Error;

        publishStateChange(old_state, new_state,
                           success ? "Training completed successfully" : "Training failed");
        setState(new_state);

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

    void TrainerManager::setEventBus(std::shared_ptr<EventBus> event_bus) {
        event_bus_ = event_bus;

        if (event_bus_) {
            // Subscribe to state query requests
            event_bus_->subscribe<QueryTrainerStateRequest>(
                [this](const QueryTrainerStateRequest& request) {
                    handleStateQueryRequest(request);
                });
        }
    }

    void TrainerManager::handleStateQueryRequest(const QueryTrainerStateRequest& request) {
        if (!event_bus_)
            return;

        QueryTrainerStateResponse response;

        // Map internal state to response state
        switch (state_.load()) {
        case State::Idle: response.state = QueryTrainerStateResponse::State::Idle; break;
        case State::Ready: response.state = QueryTrainerStateResponse::State::Ready; break;
        case State::Running: response.state = QueryTrainerStateResponse::State::Running; break;
        case State::Paused: response.state = QueryTrainerStateResponse::State::Paused; break;
        case State::Stopping: response.state = QueryTrainerStateResponse::State::Stopping; break;
        case State::Completed: response.state = QueryTrainerStateResponse::State::Completed; break;
        case State::Error: response.state = QueryTrainerStateResponse::State::Error; break;
        }

        response.current_iteration = getCurrentIteration();
        response.current_loss = getCurrentLoss();
        response.total_iterations = getTotalIterations();

        if (!last_error_.empty()) {
            response.error_message = last_error_;
        }

        event_bus_->publish(response);
    }

    // Event publishing methods
    void TrainerManager::publishStateChange(State old_state, State new_state, const std::string& reason) {
        if (!event_bus_)
            return;

        TrainerStateChangedEvent event;

        // Map states
        auto mapState = [](State s) -> QueryTrainerStateResponse::State {
            switch (s) {
            case State::Idle: return QueryTrainerStateResponse::State::Idle;
            case State::Ready: return QueryTrainerStateResponse::State::Ready;
            case State::Running: return QueryTrainerStateResponse::State::Running;
            case State::Paused: return QueryTrainerStateResponse::State::Paused;
            case State::Stopping: return QueryTrainerStateResponse::State::Stopping;
            case State::Completed: return QueryTrainerStateResponse::State::Completed;
            case State::Error: return QueryTrainerStateResponse::State::Error;
            default: return QueryTrainerStateResponse::State::Idle;
            }
        };

        event.old_state = mapState(old_state);
        event.new_state = mapState(new_state);
        event.reason = reason;

        event_bus_->publish(event);
    }

    void TrainerManager::publishTrainingStarted(int total_iterations) {
        if (event_bus_) {
            event_bus_->publish(TrainingStartedEvent{total_iterations});
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