#pragma once

#include "core/splat_data.hpp"
#include "core/trainer.hpp"
#include "visualizer/rendering_pipeline.hpp"
#include <memory>

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

        // Get model for rendering
        const SplatData* getModel() const {
            if (mode_ == Mode::Viewing && model_) {
                return model_.get();
            } else if (mode_ == Mode::Training && trainer_) {
                return &trainer_->get_strategy().get_model();
            }
            return nullptr;
        }

        // Get mutable model (no lock needed - caller handles locking)
        SplatData* getMutableModel() {
            if (mode_ == Mode::Viewing && model_) {
                return model_.get();
            } else if (mode_ == Mode::Training && trainer_) {
                return const_cast<SplatData*>(&trainer_->get_strategy().get_model());
            }
            return nullptr;
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

        // Rendering
        std::unique_ptr<RenderingPipeline> pipeline_;
    };

} // namespace gs