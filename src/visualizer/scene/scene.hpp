#pragma once

#include "core/events.hpp"
#include "core/imodel_provider.hpp"
#include "core/trainer.hpp"
#include "rendering/rendering_pipeline.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gs {

    class Scene {
    public:
        // Scene can be empty, contain models for viewing, or be linked to training
        enum class Mode {
            Empty,
            Viewing, // Static model viewing (can have multiple models)
            Training // Live training visualization (single model only)
        };

        // Model entry for multi-model support
        struct ModelEntry {
            std::string id;             // Unique identifier
            std::string name;           // Display name
            std::filesystem::path path; // Source path
            std::shared_ptr<IModelProvider> provider;
            bool visible = true;
            bool selected = false;
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

        // Single model management (for compatibility and training mode)
        void setModelProvider(std::shared_ptr<IModelProvider> provider);
        void clearModel();
        bool hasModel() const;

        // Multi-model management
        std::string addModel(const std::string& name,
                             const std::filesystem::path& path,
                             std::shared_ptr<IModelProvider> provider);
        bool removeModel(const std::string& id);
        void clearAllModels();
        size_t getModelCount() const { return models_.size(); }
        std::vector<const ModelEntry*> getModels() const;
        ModelEntry* getModel(const std::string& id);
        const ModelEntry* getModel(const std::string& id) const;

        // Selection management
        void selectModel(const std::string& id, bool exclusive = true);
        void deselectModel(const std::string& id);
        void deselectAllModels();
        std::vector<std::string> getSelectedModelIds() const;

        // Visibility management
        void setModelVisible(const std::string& id, bool visible);
        bool isModelVisible(const std::string& id) const;

        // Get combined model for rendering (merges all visible models)
        const SplatData* getModel() const;

        // Get mutable model (only for single model/training mode)
        SplatData* getMutableModel() {
            if (mode_ == Mode::Training && !models_.empty()) {
                return models_.begin()->second.provider->getMutableModel();
            }
            return nullptr;
        }

        // Convenience methods for setting specific types
        void setStandaloneModel(std::unique_ptr<SplatData> model);
        void linkToTrainer(Trainer* trainer);
        void unlinkFromTrainer();

        // Get model provider for type checking
        std::shared_ptr<IModelProvider> getModelProvider() const {
            if (!models_.empty()) {
                return models_.begin()->second.provider;
            }
            return nullptr;
        }

        // Rendering
        RenderingPipeline::RenderResult render(const RenderingPipeline::RenderRequest& request);

        // Get total gaussian count across all visible models
        size_t getTotalGaussianCount() const;

    private:
        Mode mode_ = Mode::Empty;
        std::unordered_map<std::string, ModelEntry> models_;
        std::unique_ptr<RenderingPipeline> pipeline_;
        size_t next_model_id_ = 1;

        // Merged model for multi-model rendering
        mutable std::unique_ptr<SplatData> merged_model_;
        mutable bool merged_model_dirty_ = true;

        // Generate unique model ID
        std::string generateModelId();

        // Update merged model from visible models
        void updateMergedModel();

        // Event handlers
        void handleModelInfoQuery();
        void publishModeChange(Mode old_mode, Mode new_mode);
        void publishModelAdded(const std::string& id, const ModelEntry& entry);
        void publishModelRemoved(const std::string& id);
        void publishSelectionChanged();
    };

} // namespace gs