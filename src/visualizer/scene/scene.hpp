// scene.hpp
#pragma once

#include "core/splat_data.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

namespace gs {

    class Scene {
    public:
        struct Node {
            std::string name;
            std::unique_ptr<SplatData> model;
            glm::mat4 transform{1.0f};
            bool visible = true;
            size_t gaussian_count = 0;
        };

        Scene() = default;
        ~Scene() = default;

        // Delete copy operations
        Scene(const Scene&) = delete;
        Scene& operator=(const Scene&) = delete;

        // Allow move operations
        Scene(Scene&&) = default;
        Scene& operator=(Scene&&) = default;

        // Node management
        void addNode(const std::string& name, std::unique_ptr<SplatData> model);
        void removeNode(const std::string& name);
        void setNodeVisibility(const std::string& name, bool visible);
        void clear();

        // Get combined model for rendering
        const SplatData* getCombinedModel() const;

        // Direct queries
        size_t getNodeCount() const { return nodes_.size(); }
        size_t getTotalGaussianCount() const;
        std::vector<const Node*> getNodes() const;
        const Node* getNode(const std::string& name) const;
        Node* getMutableNode(const std::string& name);
        bool hasNodes() const { return !nodes_.empty(); }

    private:
        std::vector<Node> nodes_;

        // Caching for combined model
        mutable std::unique_ptr<SplatData> cached_combined_;
        mutable bool cache_valid_ = false;

        void invalidateCache() { cache_valid_ = false; }
        void rebuildCacheIfNeeded() const;
    };

} // namespace gs
