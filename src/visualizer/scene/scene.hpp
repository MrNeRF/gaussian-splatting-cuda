#pragma once

#include "core/events.hpp"
#include "core/imodel_provider.hpp"
#include "core/trainer.hpp"
#include "rendering/rendering_pipeline.hpp"
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

        // Model management via providers
        void setModelProvider(std::shared_ptr<IModelProvider> provider);
        void clearModel();
        bool hasModel() const;

        // Get model for rendering
        const SplatData* getModel() const {
            if (model_provider_) {
                return model_provider_->getModel();
            }
            return nullptr;
        }

        // Get mutable model (no lock needed - caller handles locking)
        SplatData* getMutableModel() {
            if (model_provider_) {
                return model_provider_->getMutableModel();
            }
            return nullptr;
        }

        // Convenience methods for setting specific types
        void setStandaloneModel(std::unique_ptr<SplatData> model);
        void linkToTrainer(Trainer* trainer);
        void unlinkFromTrainer();

        // Get model provider for type checking
        std::shared_ptr<IModelProvider> getModelProvider() const {
            return model_provider_;
        }

        // Rendering
        RenderingPipeline::RenderResult render(const RenderingPipeline::RenderRequest& request);

    private:
        Mode mode_ = Mode::Empty;
        std::shared_ptr<IModelProvider> model_provider_;
        std::unique_ptr<RenderingPipeline> pipeline_;

        // Event handlers
        void handleModelInfoQuery();
        void publishModeChange(Mode old_mode, Mode new_mode);
    };

} // namespace gs