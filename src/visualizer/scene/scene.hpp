#pragma once

#include "core/imodel_provider.hpp"
#include "core/trainer.hpp"
#include "rendering/rendering.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

namespace gs {

    class Scene {
    public:
        // Scene can be empty, contain models for viewing, or be linked to training
        enum class Mode {
            Empty,
            Viewing, // Static model viewing (now supports multiple PLYs)
            Training // Live training visualization
        };

        struct SceneNode {
            std::string name;
            std::unique_ptr<SplatData> model;
            glm::mat4 transform{1.0f};
            bool visible = true;
            size_t gaussian_count = 0;
        };

        // Direct query struct
        struct ModelInfo {
            bool has_model = false;
            size_t num_gaussians = 0;
            int sh_degree = 0;
            float scene_scale = 0.0f;
            std::string source;
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

        // Multi-PLY management
        void addPLY(const std::string& name, std::unique_ptr<SplatData> model);
        void removePLY(const std::string& name);
        void clearPLYs();
        void setPLYVisibility(const std::string& name, bool visible);
        std::vector<const SceneNode*> getSceneNodes() const;
        size_t getTotalGaussianCount() const;

        // Legacy single model support for backwards compatibility
        void setModelProvider(std::shared_ptr<IModelProvider> provider);
        void clearModel();
        bool hasModel() const;

        // Get model for rendering (returns combined model in PLY mode)
        const SplatData* getModel() const;
        SplatData* getMutableModel();

        // Convenience methods for setting specific types
        void setStandaloneModel(std::unique_ptr<SplatData> model);
        void linkToTrainer(Trainer* trainer);
        void unlinkFromTrainer();

        // Get model provider for type checking
        std::shared_ptr<IModelProvider> getModelProvider() const {
            return model_provider_;
        }

        // Direct query method (replaces query event)
        ModelInfo getModelInfo() const;

    private:
        Mode mode_ = Mode::Empty;

        // For PLY viewing mode - scene graph
        std::vector<SceneNode> scene_graph_;

        // For training mode - single model provider
        std::shared_ptr<IModelProvider> model_provider_;

        // Caching for combined model
        mutable std::unique_ptr<SplatData> cached_combined_model_;
        mutable bool cache_valid_ = false;

        // Track if pipeline needs reset
        mutable bool pipeline_needs_reset_ = false;

        // Event handlers
        void publishModeChange(Mode old_mode, Mode new_mode);
        void setupEventHandlers();

        // Helper to rebuild combined model when needed
        void rebuildCombinedModelIfNeeded() const;
    };

} // namespace gs