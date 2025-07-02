#pragma once

#include "core/istrategy.hpp"
#include "core/splat_data.hpp"
#include "core/newton_optimizer.hpp"
#include "core/parameters.hpp"
#include "core/camera_dataset.hpp" // For CameraDataset, Camera
#include "core/rasterizer.hpp"   // For gs::RenderOutput

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory> // For std::unique_ptr

namespace gs { // Forward declare RenderOutput if not fully included via rasterizer.hpp
    struct RenderOutput;
}

class NewtonStrategy : public IStrategy {
public:
    NewtonStrategy(
        const gs::param::TrainingParameters& training_params,
        SplatData& initial_splat_data, // Takes ownership or reference? Let's assume reference for now.
                                       // Or, strategy creates its own SplatData.
                                       // For consistency with how Trainer likely works, it might receive it.
                                       // However, typical strategies own their model. Let's make it own.
        std::shared_ptr<CameraDataset> train_dataset_for_knn // Used for KNN search
    );

    ~NewtonStrategy() override = default;

    void initialize(const gs::param::OptimizationParameters& optimParams) override;

    // Called after loss.backward() but before optimizer.step()
    // We'll use this to store data needed by NewtonOptimizer::step
    void post_backward(int iter, gs::RenderOutput& render_output) override;

    // Main optimization step
    void step(int iter) override;

    bool is_refining(int iter) const override;

    SplatData& get_model() override { return *splat_data_; }
    const SplatData& get_model() const override { return *splat_data_; }

    // Method for Trainer to set the current view's data if post_backward isn't enough
    // (though post_backward should be sufficient if we cache the data there)
    void set_current_view_data(
        const Camera* primary_camera,
        const torch::Tensor& primary_gt_image, // on device
        const gs::RenderOutput& render_output, // from rasterizer
        const gs::param::OptimizationParameters& opt_params,
        int iteration
        );

private:
    std::unique_ptr<SplatData> splat_data_; // Strategy owns the model
    std::unique_ptr<NewtonOptimizer> optimizer_;
    gs::param::OptimizationParameters optim_params_cache_; // Store a copy of optimization params

    // Data cached from post_backward/set_current_view_data for NewtonOptimizer::step
    int current_iter_ = 0;
    const Camera* current_primary_camera_ = nullptr; // Raw pointer, lifetime managed by CameraDataset in Trainer
    torch::Tensor current_primary_gt_image_;     // Copied tensor
    gs::RenderOutput current_render_output_cache_; // Copied struct
    torch::Tensor current_visibility_mask_for_model_; // Mask for all P_total Gaussians

    // For KNN logic:
    std::shared_ptr<CameraDataset> train_dataset_ref_; // For accessing all camera poses and loading GTs
    std::vector<const Camera*> all_train_cameras_cache_; // Cache of camera pointers from train_dataset_ref_

    Eigen::Vector3f scene_center_for_knn_;
    float scene_radius_for_knn_;
    std::vector<Eigen::Vector3f> projected_camera_positions_on_sphere_;
    std::vector<std::pair<const Camera*, torch::Tensor>> current_knn_targets_gpu_; // GT images on GPU

    void initialize_knn_data_if_needed();
    void find_knn_for_current_primary(const Camera* primary_cam_in);

    // Helper to calculate the visibility mask for all P_total gaussians
    // based on the render_output.visibility (which is for P_render gaussians)
    // and a mapping from P_render to P_total (ranks).
    // This is complex and needs the 'ranks' tensor. For now, placeholder.
    void compute_visibility_mask_for_model(const gs::RenderOutput& render_output, const SplatData& model);
};
