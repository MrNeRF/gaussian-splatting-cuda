// scene.cpp
#include "scene/scene.hpp"
#include <algorithm>
#include <print>
#include <torch/torch.h>

namespace gs {

    void Scene::addNode(const std::string& name, std::unique_ptr<SplatData> model) {
        // Calculate gaussian count before moving
        size_t gaussian_count = static_cast<size_t>(model->size());

        // Check if name already exists
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });

        if (it != nodes_.end()) {
            // Replace existing
            it->model = std::move(model);
            it->gaussian_count = gaussian_count;
        } else {
            // Add new node
            Node node{
                .name = name,
                .model = std::move(model),
                .transform = glm::mat4(1.0f),
                .visible = true,
                .gaussian_count = gaussian_count};
            nodes_.push_back(std::move(node));
        }

        invalidateCache();
        std::println("Scene: Added node '{}' with {} gaussians", name, gaussian_count);
    }

    void Scene::removeNode(const std::string& name) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });

        if (it != nodes_.end()) {
            nodes_.erase(it);
            invalidateCache();
            std::println("Scene: Removed node '{}'", name);
        }
    }

    void Scene::setNodeVisibility(const std::string& name, bool visible) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });

        if (it != nodes_.end() && it->visible != visible) {
            it->visible = visible;
            invalidateCache();
        }
    }

    void Scene::clear() {
        nodes_.clear();
        cached_combined_.reset();
        cache_valid_ = false;
    }

    const SplatData* Scene::getCombinedModel() const {
        rebuildCacheIfNeeded();
        return cached_combined_.get();
    }

    size_t Scene::getTotalGaussianCount() const {
        size_t total = 0;
        for (const auto& node : nodes_) {
            if (node.visible) {
                total += node.gaussian_count;
            }
        }
        return total;
    }

    std::vector<const Scene::Node*> Scene::getNodes() const {
        std::vector<const Node*> result;
        result.reserve(nodes_.size());
        for (const auto& node : nodes_) {
            result.push_back(&node);
        }
        return result;
    }

    const Scene::Node* Scene::getNode(const std::string& name) const {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });
        return (it != nodes_.end()) ? &(*it) : nullptr;
    }

    Scene::Node* Scene::getMutableNode(const std::string& name) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });
        if (it != nodes_.end()) {
            invalidateCache();
            return &(*it);
        }
        return nullptr;
    }

    void Scene::rebuildCacheIfNeeded() const {
        if (cache_valid_) {
            return;
        }

        // Collect visible models
        std::vector<const SplatData*> visible_models;
        size_t total_gaussians = 0;

        for (const auto& node : nodes_) {
            if (node.visible && node.model) {
                visible_models.push_back(node.model.get());
                total_gaussians += node.gaussian_count;
            }
        }

        if (visible_models.empty()) {
            cached_combined_.reset();
            cache_valid_ = true;
            return;
        }

        std::println("Scene: Rebuilding combined model with {} visible models, {} total gaussians",
                     visible_models.size(), total_gaussians);

        // Get device and dtype from first model
        auto device = visible_models[0]->means().device();
        auto dtype = visible_models[0]->means().dtype();

        // Create tensor options with both device and dtype
        auto opts = torch::TensorOptions().dtype(dtype).device(device);

        // Pre-allocate tensors on the target device
        auto combined_means = torch::empty({static_cast<int64_t>(total_gaussians), 3}, opts);
        auto combined_sh0 = torch::empty({static_cast<int64_t>(total_gaussians),
                                          visible_models[0]->sh0().size(1),
                                          visible_models[0]->sh0().size(2)},
                                         opts);
        auto combined_shN = torch::empty({static_cast<int64_t>(total_gaussians),
                                          visible_models[0]->shN().size(1),
                                          visible_models[0]->shN().size(2)},
                                         opts);
        auto combined_opacity = torch::empty({static_cast<int64_t>(total_gaussians), 1}, opts);
        auto combined_scaling = torch::empty({static_cast<int64_t>(total_gaussians), 3}, opts);
        auto combined_rotation = torch::empty({static_cast<int64_t>(total_gaussians), 4}, opts);

        // Concatenate all visible models
        size_t current_idx = 0;
        int max_sh_degree = 0;
        float avg_scene_scale = 0.0f;

        for (const auto* model : visible_models) {
            int64_t model_size = model->size();

            // Copy data
            combined_means.index({torch::indexing::Slice(current_idx, current_idx + model_size)}) = model->means();
            combined_sh0.index({torch::indexing::Slice(current_idx, current_idx + model_size)}) = model->sh0();
            combined_shN.index({torch::indexing::Slice(current_idx, current_idx + model_size)}) = model->shN();
            combined_opacity.index({torch::indexing::Slice(current_idx, current_idx + model_size)}) = model->opacity_raw();
            combined_scaling.index({torch::indexing::Slice(current_idx, current_idx + model_size)}) = model->scaling_raw();
            combined_rotation.index({torch::indexing::Slice(current_idx, current_idx + model_size)}) = model->rotation_raw();

            max_sh_degree = std::max(max_sh_degree, model->get_active_sh_degree());
            avg_scene_scale += model->get_scene_scale();
            current_idx += model_size;
        }

        avg_scene_scale /= visible_models.size();

        cached_combined_ = std::make_unique<SplatData>(
            max_sh_degree,
            combined_means,
            combined_sh0,
            combined_shN,
            combined_scaling,
            combined_rotation,
            combined_opacity,
            avg_scene_scale);

        cache_valid_ = true;
    }

} // namespace gs
