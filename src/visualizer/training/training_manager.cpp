#include "training/training_manager.hpp"
#include <print>

namespace gs {

    TrainerManager::TrainerManager() {
        setupEventHandlers();
    }

    TrainerManager::~TrainerManager() {
        // Ensure training is stopped before destruction
        if (training_thread_ && training_thread_->joinable()) {
            std::println("TrainerManager: Stopping training thread...");
            stopTraining();
            waitForCompletion();
        }
    }

    void TrainerManager::setupEventHandlers() {
        using namespace events::query;

        // Handle trainer state queries
        GetTrainerState::when([this](const auto&) {
            TrainerState response;

            // Map internal state to response state
            switch (state_.load()) {
            case State::Idle:
                response.state = TrainerState::State::Idle;
                break;
            case State::Ready:
                response.state = TrainerState::State::Ready;
                break;
            case State::Running:
                response.state = TrainerState::State::Running;
                break;
            case State::Paused:
                response.state = TrainerState::State::Paused;
                break;
            case State::Completed:
                response.state = TrainerState::State::Completed;
                break;
            case State::Error:
                response.state = TrainerState::State::Error;
                break;
            default:
                response.state = TrainerState::State::Idle;
            }

            response.current_iteration = getCurrentIteration();
            response.current_loss = getCurrentLoss();
            response.total_iterations = getTotalIterations();

            if (!last_error_.empty()) {
                response.error_message = last_error_;
            }

            response.emit();
        });
    }

    void TrainerManager::setTrainer(std::unique_ptr<Trainer> trainer) {
        // Clear any existing trainer first
        clearTrainer();

        if (trainer) {
            trainer_ = std::move(trainer);
            setState(State::Ready);

            // Trainer is ready
            events::internal::TrainerReady{}.emit();
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

        setState(State::Running);

        // Emit training started event
        events::state::TrainingStarted{
            .total_iterations = getTotalIterations()}
            .emit();

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
            setState(State::Paused);

            events::state::TrainingPaused{
                .iteration = getCurrentIteration()}
                .emit();

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

            events::state::TrainingResumed{
                .iteration = getCurrentIteration()}
                .emit();

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

        events::state::TrainingStopped{
            .iteration = getCurrentIteration(),
            .user_requested = true}
            .emit();
    }

    void TrainerManager::requestSaveCheckpoint() {
        if (trainer_ && isTrainingActive()) {
            trainer_->request_save();
            std::println("TrainerManager: Checkpoint save requested");

            events::notify::Log{
                .level = events::notify::Log::Level::Info,
                .message = "Checkpoint save requested",
                .source = "TrainerManager"}
                .emit();
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

        setState(success ? State::Completed : State::Error);

        events::state::TrainingCompleted{
            .iteration = getCurrentIteration(),
            .final_loss = getCurrentLoss(),
            .success = success,
            .error = error.empty() ? std::nullopt : std::optional(error)}
            .emit();

        // Notify completion
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            training_complete_ = true;
        }
        completion_cv_.notify_all();
    }

} // namespace gs