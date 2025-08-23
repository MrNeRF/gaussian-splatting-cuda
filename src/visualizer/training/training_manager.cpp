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

    void TrainerManager::setTrainer(std::unique_ptr<Trainer> trainer) {
        // Clear any existing trainer first
        clearTrainer();

        if (trainer) {
            trainer_ = std::move(trainer);
            trainer_->setProject(project_);

            setState(State::Ready);

            // Trainer is ready
            events::internal::TrainerReady{}.emit();
        }
    }

    bool TrainerManager::hasTrainer() const {
        return trainer_ != nullptr;
    }

    void TrainerManager::clearTrainer() {
        events::cmd::StopTraining{}.emit();
        // Stop any ongoing training first
        if (isTrainingActive()) {
            std::println("TrainerManager: Stopping training before clearing trainer...");
            stopTraining();
            waitForCompletion();
        }

        // Additional safety: ensure thread is properly stopped even if not "active"
        if (training_thread_ && training_thread_->joinable()) {
            std::println("TrainerManager: Force stopping training thread...");
            training_thread_->request_stop();

            // Try to wait for completion with a short timeout
            auto timeout = std::chrono::milliseconds(500);
            {
                std::unique_lock<std::mutex> lock(completion_mutex_);
                if (completion_cv_.wait_for(lock, timeout, [this] { return training_complete_; })) {
                    lock.unlock();
                    std::println("TrainerManager: Thread completed gracefully, joining...");
                    training_thread_->join();
                } else {
                    lock.unlock();
                    std::println("TrainerManager: Thread didn't respond to stop request, detaching...");
                    training_thread_->detach();
                }
            }
            training_thread_.reset();
        }

        // Now safe to clear the trainer
        trainer_.reset();
        last_error_.clear();
        setState(State::Idle);

        // Reset loss buffer
        loss_buffer_.clear();
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
        return trainer_->getParams().optimization.iterations;
    }

    int TrainerManager::getNumSplats() const {
        if (!trainer_)
            return 0;
        return static_cast<int>(trainer_->get_strategy().get_model().size());
    }

    void TrainerManager::updateLoss(float loss) {
        std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
        loss_buffer_.push_back(loss);
        while (loss_buffer_.size() > static_cast<size_t>(max_loss_points_)) {
            loss_buffer_.pop_front();
        }
    }

    std::deque<float> TrainerManager::getLossBuffer() const {
        std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
        return loss_buffer_;
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

    void TrainerManager::setupEventHandlers() {
        using namespace events;

        // Listen for training progress events - only update loss buffer
        state::TrainingProgress::when([this](const auto& event) {
            updateLoss(event.loss);
        });
    }

    std::shared_ptr<const Camera> TrainerManager::getCamById(int camId) const {
        if (trainer_) {
            return trainer_->getCamById(camId);
        }
        std::cerr << " getCamById trainer is not initialized " << std::endl;
        return nullptr;
    }

    std::vector<std::shared_ptr<const Camera>> TrainerManager::getCamList() const {
        if (trainer_) {
            return trainer_->getCamList();
        }
        std::cerr << " getCamList trainer is not initialized " << std::endl;
        return {};
    }

    void TrainerManager::setProject(std::shared_ptr<gs::management::Project> project) {
        project_ = project;
        if (trainer_) {
            trainer_->setProject(project);
        }
    }

} // namespace gs