// include/newton_optimizer.hpp
#pragma once

#include "core/splat_data.hpp"
#include "core/camera.hpp" // Included for Camera&
#include "core/rasterizer.hpp" // Included for RenderOutput
#include "core/parameters.hpp" // For gs::param::OptimizationParameters
#include <torch/torch.h>
#include <vector>
#include <string> // For std::string in Options

// Forward declaration
namespace gs {
    struct RenderOutput;
    namespace param {
        struct OptimizationParameters;
    }
}

class NewtonOptimizer {
public:
    struct Options {
        double step_scale = 1.0;
        double damping = 1e-6;
        int knn_k = 3;
        float secondary_target_downsample = 0.5;
        float lambda_dssim_for_hessian = 0.2f; // Assuming similar to main loss lambda

        bool optimize_means = true;
        bool optimize_scales = true;
        bool optimize_rotations = true;
        bool optimize_opacities = true;
        bool optimize_shs = true;

        // For L2 loss part in Hessian computation (as per paper)
        bool use_l2_for_hessian_L_term = true;
        bool debug_print_shapes = false;         // Enable debug prints for tensor shapes
    };

    NewtonOptimizer(SplatData& splat_data,
                    const gs::param::OptimizationParameters& opt_params,
                    Options options = {});

    void step(int iteration,
              const torch::Tensor& visibility_mask, // Mask for model_.means()
              const gs::RenderOutput& current_render_output, // from primary target
              const Camera& primary_camera,
              const torch::Tensor& primary_gt_image, // Already on device
              const std::vector<std::pair<const Camera*, torch::Tensor>>& knn_secondary_targets_data
              );

private:
    SplatData& model_;
    const gs::param::OptimizationParameters& opt_params_ref_;
    Options options_;

    // --- Loss Derivatives ---
    struct LossDerivatives {
        torch::Tensor dL_dc;     // [H, W, 3] or [B, H, W, 3]
        torch::Tensor d2L_dc2_diag; // [H, W, 3] or [B, H, W, 3] (diagonal of the 3x3 Hessian block for each pixel)
    };
    LossDerivatives compute_loss_derivatives_cuda(
        const torch::Tensor& rendered_image, // [H, W, 3]
        const torch::Tensor& gt_image,       // [H, W, 3]
        float lambda_dssim,
        bool use_l2_loss_term
    );


    // --- Position (Means) ---
    struct PositionHessianOutput {
        torch::Tensor H_p_packed; // Packed symmetric 3x3 Hessian per Gaussian [N_vis, 6]
        torch::Tensor grad_p;     // Gradient 𝜕L/𝜕p per Gaussian [N_vis, 3]
    };
    PositionHessianOutput compute_position_hessian_components_cuda(
        // Pass necessary parts of model, camera, render_output, loss_derivs
        // and visibility information to select/process only visible Gaussians.
        const SplatData& model_snapshot, // A const reference to current model state
        const torch::Tensor& visibility_mask_for_model, // [Total_N] bool tensor
        const Camera& camera,
        const gs::RenderOutput& render_output, // Contains means2d, etc. for culled set by rasterizer
        const LossDerivatives& loss_derivs,
        int num_visible_gaussians_in_total_model // Count of true in visibility_mask_for_model
    );

    torch::Tensor compute_projected_position_hessian_and_gradient(
        const torch::Tensor& H_p_packed, // [N_vis, 6]
        const torch::Tensor& grad_p,     // [N_vis, 3]
        const torch::Tensor& means_3d_visible, // [N_vis, 3] (only visible means from model)
        const Camera& camera,
        torch::Tensor& out_grad_v       // Output for projected gradient [N_vis, 2]
                                         // Returns H_v_packed [N_vis, 3] (for symmetric 2x2)
    );

    torch::Tensor solve_and_project_position_updates(
        const torch::Tensor& H_v_projected_packed, // [N_vis, 3]
        const torch::Tensor& grad_v_projected,     // [N_vis, 2]
        const torch::Tensor& means_3d_visible,     // [N_vis, 3]
        const Camera& camera,
        double damping,
        double step_scale
    ); // Returns delta_p [N_vis, 3]

    // Placeholder for other parameter groups
    // These will be the actual Newton update computation functions for each attribute
    struct AttributeUpdateOutput {
        torch::Tensor delta;
        bool success = true;
        // Default constructor for placeholder returns
        AttributeUpdateOutput(torch::Tensor d = torch::empty({0}), bool s = true) : delta(d), success(s) {}
    };

    AttributeUpdateOutput compute_scale_updates_newton(
        /* const SplatData& model_snapshot, // Or use member model_ directly */
        const torch::Tensor& visible_indices,
        const LossDerivatives& loss_derivs,
        const Camera& camera,
        const gs::RenderOutput& render_output);
        // opt_params_ref_ and options_ are member variables

    AttributeUpdateOutput compute_rotation_updates_newton(
        const torch::Tensor& visible_indices,
        const LossDerivatives& loss_derivs,
        const Camera& camera,
        const gs::RenderOutput& render_output);

    AttributeUpdateOutput compute_opacity_updates_newton(
        const torch::Tensor& visible_indices,
        const LossDerivatives& loss_derivs,
        const Camera& camera,
        const gs::RenderOutput& render_output);

    AttributeUpdateOutput compute_sh_updates_newton(
        const torch::Tensor& visible_indices,
        const LossDerivatives& loss_derivs,
        const Camera& camera,
        const gs::RenderOutput& render_output);
};
