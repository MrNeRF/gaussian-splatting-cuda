#pragma once

#include "core/splat_data.hpp"
#include "core/trainer.hpp"
#include "visualizer/rendering_pipeline.hpp"
#include <memory>
#include <mutex>

namespace gs {

    class Scene {
    public:
        // Scene can be empty, contain a model for viewing, or be linked to training
        enum class Mode {
            Empty,
            Viewing, // Static model viewing
            Training // Live training visualization
        };

        Scene();
        ~Scene() = default;

        // Delete copy operations
        Scene(const Scene&) = delete;
        Scene& operator=(const Scene&) = delete;

        // Allow move operations
        Scene(Scene&&) = default;
        Scene& operator=(Scene&&) = default;

        // Mode management
        Mode getMode() const { return mode_; }

        // Model management
        void setModel(std::unique_ptr<SplatData> model);
        void clearModel();
        bool hasModel() const;

        // Thread-safe model access for rendering
        template <typename Func>
        auto withModel(Func&& func) const -> decltype(func(std::declval<const SplatData*>())) {
            std::lock_guard<std::mutex> lock(model_mutex_);
            if (mode_ == Mode::Viewing && model_) {
                return func(model_.get());
            } else if (mode_ == Mode::Training && trainer_) {
                return func(&trainer_->get_strategy().get_model());
            }
            return func(nullptr);
        }

        // Thread-safe mutable model access
        template <typename Func>
        auto withMutableModel(Func&& func) -> decltype(func(std::declval<SplatData*>())) {
            std::lock_guard<std::mutex> lock(model_mutex_);
            if (mode_ == Mode::Viewing && model_) {
                return func(model_.get());
            } else if (mode_ == Mode::Training && trainer_) {
                // Cast away const for training mode (trainer owns the model)
                return func(const_cast<SplatData*>(&trainer_->get_strategy().get_model()));
            }
            return func(nullptr);
        }

        // Rendering
        RenderingPipeline::RenderResult render(const RenderingPipeline::RenderRequest& request);

        // For training mode - just store a reference
        void linkToTrainer(Trainer* trainer);
        void unlinkFromTrainer();

        // Get trainer (for GUI access)
        Trainer* getTrainer() const { return trainer_; }

        // Get standalone model (for GUI access)
        SplatData* getStandaloneModel() const {
            return (mode_ == Mode::Viewing) ? model_.get() : nullptr;
        }

    private:
        Mode mode_ = Mode::Empty;

        // Model ownership for viewing mode
        std::unique_ptr<SplatData> model_;

        // Reference for training mode
        Trainer* trainer_ = nullptr;

        // Thread safety
        mutable std::mutex model_mutex_;

        // Rendering
        std::unique_ptr<RenderingPipeline> pipeline_;
    };

} // namespace gs
