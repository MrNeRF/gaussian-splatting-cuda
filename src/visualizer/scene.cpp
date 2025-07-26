#include "visualizer/scene.hpp"
#include <print>

namespace gs {

    Scene::Scene()
        : pipeline_(std::make_unique<RenderingPipeline>()) {
    }

    void Scene::setModel(std::unique_ptr<SplatData> model) {
        std::lock_guard<std::mutex> lock(model_mutex_);

        // Clear any training link
        trainer_ = nullptr;

        // Set the new model
        model_ = std::move(model);
        mode_ = model_ ? Mode::Viewing : Mode::Empty;
    }

    void Scene::clearModel() {
        std::lock_guard<std::mutex> lock(model_mutex_);

        model_.reset();
        trainer_ = nullptr;
        mode_ = Mode::Empty;
    }

    bool Scene::hasModel() const {
        std::lock_guard<std::mutex> lock(model_mutex_);
        return (mode_ != Mode::Empty);
    }

    void Scene::linkToTrainer(Trainer* trainer) {
        std::lock_guard<std::mutex> lock(model_mutex_);

        // Clear any viewing model
        model_.reset();

        // Link to trainer
        trainer_ = trainer;
        mode_ = trainer ? Mode::Training : Mode::Empty;
    }

    void Scene::unlinkFromTrainer() {
        std::lock_guard<std::mutex> lock(model_mutex_);

        trainer_ = nullptr;
        if (!model_) {
            mode_ = Mode::Empty;
        }
    }

    RenderingPipeline::RenderResult Scene::render(const RenderingPipeline::RenderRequest& request) {
        RenderingPipeline::RenderResult result;
        result.valid = false;

        // Use withModel to safely access the model
        withModel([&](const SplatData* model) {
            if (model && pipeline_) {
                result = pipeline_->render(*model, request);
            }
            return 0; // Return value doesn't matter, just need something
        });

        return result;
    }

} // namespace gs
