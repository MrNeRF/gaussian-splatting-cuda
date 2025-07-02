// src/newton_optimizer.cpp
#include "newton_optimizer.hpp" // Moved from core/
#include "kernels/newton_kernels.cuh" // Path relative to include paths, or needs adjustment
#include "core/torch_utils.hpp" // Assuming torch_utils is still in core/

// Constructor
NewtonOptimizer::NewtonOptimizer(SplatData& splat_data,
                                 const gs::param::OptimizationParameters& opt_params,
                                 Options options)
    : model_(splat_data), opt_params_ref_(opt_params), options_(options) {
    // TODO: Initialization if needed, e.g. pre-allocate tensors if sizes are fixed
}

// --- Loss Derivatives ---
NewtonOptimizer::LossDerivatives NewtonOptimizer::compute_loss_derivatives_cuda(
    const torch::Tensor& rendered_image,
    const torch::Tensor& gt_image,
    float lambda_dssim,
    bool use_l2_loss_term) {

    TORCH_CHECK(rendered_image.device().is_cuda(), "rendered_image must be a CUDA tensor");
    TORCH_CHECK(gt_image.device().is_cuda(), "gt_image must be a CUDA tensor");
    TORCH_CHECK(rendered_image.sizes() == gt_image.sizes(), "rendered_image and gt_image must have the same size");
    TORCH_CHECK(rendered_image.dim() == 3 && rendered_image.size(2) == 3, "Images must be HxWx3, got ", rendered_image.sizes());

    auto tensor_options = torch::TensorOptions().device(rendered_image.device()).dtype(rendered_image.dtype());
    torch::Tensor dL_dc = torch::zeros_like(rendered_image, tensor_options);
    torch::Tensor d2L_dc2_diag = torch::zeros_like(rendered_image, tensor_options);

    NewtonKernels::compute_loss_derivatives_kernel_launcher(
        rendered_image, gt_image, lambda_dssim, use_l2_loss_term,
        dL_dc, d2L_dc2_diag
    );

    return {dL_dc, d2L_dc2_diag};
}

// --- Position (Means) ---
NewtonOptimizer::PositionHessianOutput NewtonOptimizer::compute_position_hessian_components_cuda(
    const SplatData& model_snapshot,
    const torch::Tensor& visibility_mask_for_model,
    const Camera& camera,
    const gs::RenderOutput& render_output, // Contains data for rasterizer-culled Gaussians
    const LossDerivatives& loss_derivs,
    int num_visible_gaussians_in_total_model // Number of Gaussians to produce output for
) {
    // Use const getter for SplatData when model_snapshot is const
    torch::Tensor means_tensor = model_snapshot.get_means();
    auto dev = means_tensor.device();
    auto dtype = means_tensor.dtype();
    auto tensor_opts = torch::TensorOptions().device(dev).dtype(dtype);

    // Output tensors for the *num_visible_gaussians_in_total_model*
    torch::Tensor H_p_output_packed = torch::zeros({num_visible_gaussians_in_total_model, 6}, tensor_opts);
    torch::Tensor grad_p_output = torch::zeros({num_visible_gaussians_in_total_model, 3}, tensor_opts);

    // Prepare camera parameters
    torch::Tensor view_mat_tensor = camera.world_view_transform().to(dev).to(dtype); // Corrected method
    torch::Tensor K_matrix = camera.K().to(dev).to(dtype); // Corrected method

    // Compute camera center C_w = -R_wc^T * t_wc from world_view_transform V = [R_wc | t_wc]
    // V is typically [4,4] or [3,4]. Assuming [4,4] world-to-camera.
    torch::Tensor R_wc = view_mat_tensor.slice(0, 0, 3).slice(1, 0, 3); // Top-left 3x3
    torch::Tensor t_wc = view_mat_tensor.slice(0, 0, 3).slice(1, 3, 4); // Top-right 3x1
    torch::Tensor cam_pos_tensor = -torch::matmul(R_wc.transpose(0, 1), t_wc).squeeze(); // Ensure it's [3]
    if (cam_pos_tensor.dim() > 1) cam_pos_tensor = cam_pos_tensor.squeeze();


    // The kernel needs to map RenderOutput's culled set of Gaussians (means2d, depths, radii)
    // back to the original model's Gaussians, or use the visibility_mask_for_model.
    // This is a complex part of the kernel design.
    // For now, we pass what we have. The kernel must be robust.
    // `render_output.visibility_indices` could be a map from render_output's internal indexing to original model indices.
    // `render_output.visibility_filter` could be a boolean mask on the *culled* set from rasterizer.

    NewtonKernels::compute_position_hessian_components_kernel_launcher(
        render_output.height, render_output.width, render_output.image.size(-1), // Image: H, W, C
        static_cast<int>(model_snapshot.size()), // Total P Gaussians in model
        gs::torch_utils::get_const_data_ptr<float>(model_snapshot.get_means()),
        gs::torch_utils::get_const_data_ptr<float>(model_snapshot.get_scaling()),
        gs::torch_utils::get_const_data_ptr<float>(model_snapshot.get_rotation()),
        gs::torch_utils::get_const_data_ptr<float>(model_snapshot.get_opacity()),
        gs::torch_utils::get_const_data_ptr<float>(model_snapshot.get_shs()),
        model_snapshot.get_active_sh_degree(),
        static_cast<int>(model_snapshot.get_shs().size(1)), // sh_coeffs_dim
        gs::torch_utils::get_const_data_ptr<float>(view_mat_tensor),
        gs::torch_utils::get_const_data_ptr<float>(K_matrix), // Was projection_matrix_for_jacobian
        gs::torch_utils::get_const_data_ptr<float>(cam_pos_tensor),
        // Data from RenderOutput (already for a culled set of Gaussians)
        gs::torch_utils::get_const_data_ptr<float>(render_output.means2d),
        gs::torch_utils::get_const_data_ptr<float>(render_output.depths),
        gs::torch_utils::get_const_data_ptr<float>(render_output.radii),
        // How to map render_output's Gaussians to original model indices or use visibility_mask_for_model
        // is critical for the kernel. The ranks tensor (mapping render_output indices to original model indices) is needed here.
        // Parameter for visibility_indices_in_render_output removed from kernel launcher.
        static_cast<int>(render_output.means2d.size(0)), // P_render: Number of Gaussians in render_output arrays
        gs::torch_utils::get_const_data_ptr<bool>(visibility_mask_for_model), // Mask on *all* model Gaussians
        gs::torch_utils::get_const_data_ptr<float>(loss_derivs.dL_dc),
        gs::torch_utils::get_const_data_ptr<float>(loss_derivs.d2L_dc2_diag),
        num_visible_gaussians_in_total_model, // Number of Gaussians to produce output for
        gs::torch_utils::get_data_ptr<float>(H_p_output_packed),
        gs::torch_utils::get_data_ptr<float>(grad_p_output)
    );

    return {H_p_output_packed, grad_p_output};
}

torch::Tensor NewtonOptimizer::compute_projected_position_hessian_and_gradient(
    const torch::Tensor& H_p_packed,
    const torch::Tensor& grad_p,
    const torch::Tensor& means_3d_visible,
    const Camera& camera,
    torch::Tensor& out_grad_v
) {
    TORCH_CHECK(H_p_packed.device().is_cuda() && grad_p.device().is_cuda() &&
                means_3d_visible.device().is_cuda() && out_grad_v.device().is_cuda(),
                "All tensors for projection must be CUDA tensors");

    int num_visible_gaussians = H_p_packed.size(0);
    TORCH_CHECK(grad_p.size(0) == num_visible_gaussians &&
                means_3d_visible.size(0) == num_visible_gaussians &&
                out_grad_v.size(0) == num_visible_gaussians, "Size mismatch in projection inputs/outputs");

    auto tensor_opts = H_p_packed.options();
    torch::Tensor H_v_packed = torch::zeros({num_visible_gaussians, 3}, tensor_opts); // 3 for symmetric 2x2

    torch::Tensor view_mat_tensor = camera.world_view_transform().to(tensor_opts.device()); // Corrected
    // Compute camera center C_w = -R_wc^T * t_wc
    torch::Tensor R_wc_proj = view_mat_tensor.slice(0, 0, 3).slice(1, 0, 3);
    torch::Tensor t_wc_proj = view_mat_tensor.slice(0, 0, 3).slice(1, 3, 4);
    torch::Tensor cam_pos_tensor = -torch::matmul(R_wc_proj.transpose(0, 1), t_wc_proj).squeeze();
    if (cam_pos_tensor.dim() > 1) cam_pos_tensor = cam_pos_tensor.squeeze();
    cam_pos_tensor = cam_pos_tensor.to(tensor_opts.device());


    NewtonKernels::project_position_hessian_gradient_kernel_launcher(
        num_visible_gaussians,
        gs::torch_utils::get_const_data_ptr<float>(H_p_packed),
        gs::torch_utils::get_const_data_ptr<float>(grad_p),
        gs::torch_utils::get_const_data_ptr<float>(means_3d_visible),
        gs::torch_utils::get_const_data_ptr<float>(view_mat_tensor),
        gs::torch_utils::get_const_data_ptr<float>(cam_pos_tensor),
        gs::torch_utils::get_data_ptr<float>(H_v_packed),
        gs::torch_utils::get_data_ptr<float>(out_grad_v)
    );
    return H_v_packed;
}

torch::Tensor NewtonOptimizer::solve_and_project_position_updates(
    const torch::Tensor& H_v_projected_packed, // [N_vis, 3]
    const torch::Tensor& grad_v_projected,     // [N_vis, 2]
    const torch::Tensor& means_3d_visible,     // [N_vis, 3]
    const Camera& camera,
    double damping,
    double step_scale
) {
    int num_visible_gaussians = H_v_projected_packed.size(0);
    auto tensor_opts = H_v_projected_packed.options();

    torch::Tensor delta_v = torch::zeros({num_visible_gaussians, 2}, tensor_opts);
    NewtonKernels::batch_solve_2x2_system_kernel_launcher(
        num_visible_gaussians,
        gs::torch_utils::get_const_data_ptr<float>(H_v_projected_packed),
        gs::torch_utils::get_const_data_ptr<float>(grad_v_projected),
        static_cast<float>(damping),
        static_cast<float>(step_scale), // step_scale is applied inside kernel: delta_v = -step_scale * H_inv * g
        gs::torch_utils::get_data_ptr<float>(delta_v)
    );

    torch::Tensor delta_p = torch::zeros({num_visible_gaussians, 3}, tensor_opts);
    torch::Tensor view_mat_tensor = camera.world_view_transform().to(tensor_opts.device()); // Corrected
    // Compute camera center C_w = -R_wc^T * t_wc
    torch::Tensor R_wc_solve = view_mat_tensor.slice(0, 0, 3).slice(1, 0, 3);
    torch::Tensor t_wc_solve = view_mat_tensor.slice(0, 0, 3).slice(1, 3, 4);
    torch::Tensor cam_pos_tensor = -torch::matmul(R_wc_solve.transpose(0, 1), t_wc_solve).squeeze();
    if (cam_pos_tensor.dim() > 1) cam_pos_tensor = cam_pos_tensor.squeeze();
    cam_pos_tensor = cam_pos_tensor.to(tensor_opts.device());

    NewtonKernels::project_update_to_3d_kernel_launcher(
        num_visible_gaussians,
        gs::torch_utils::get_const_data_ptr<float>(delta_v),
        gs::torch_utils::get_const_data_ptr<float>(means_3d_visible),
        gs::torch_utils::get_const_data_ptr<float>(view_mat_tensor),
        gs::torch_utils::get_const_data_ptr<float>(cam_pos_tensor),
        gs::torch_utils::get_data_ptr<float>(delta_p)
    );
    return delta_p;
}


// Main step function (partial implementation for position)
void NewtonOptimizer::step(int iteration,
                           const torch::Tensor& visibility_mask_for_model, // Boolean mask for model_.means() [Total_N]
                           const gs::RenderOutput& current_render_output, // From primary target
                           const Camera& primary_camera,
                           const torch::Tensor& primary_gt_image, // Already on device [H,W,C]
                           const std::vector<std::pair<const Camera*, torch::Tensor>>& knn_secondary_targets_data) {

    if (!options_.optimize_means) {
        return;
    }

    torch::NoGradGuard no_grad;

    torch::Tensor visible_indices = torch::where(visibility_mask_for_model)[0];
    int num_visible_gaussians_in_model = visible_indices.size(0);

    if (num_visible_gaussians_in_model == 0) {
        return;
    }

    torch::Tensor means_visible_from_model = model_.means().detach().index_select(0, visible_indices);

    // I. Compute Loss Derivatives for primary target
    LossDerivatives primary_loss_derivs = compute_loss_derivatives_cuda(
        current_render_output.image.squeeze(0),
        primary_gt_image,
        options_.lambda_dssim_for_hessian,
        options_.use_l2_for_hessian_L_term
    );

    // II. Compute Hessian components (H_p, g_p) for primary target
    PositionHessianOutput primary_hess_output = compute_position_hessian_components_cuda(
        model_, visibility_mask_for_model, primary_camera, current_render_output, primary_loss_derivs, num_visible_gaussians_in_model
    );

    torch::Tensor H_p_total_packed = primary_hess_output.H_p_packed.clone(); // [N_vis_model, 6]
    torch::Tensor g_p_total_visible = primary_hess_output.grad_p.clone(); // [N_vis_model, 3]

    // III. Handle Secondary Targets for Overshoot Prevention
    if (options_.knn_k > 0 && !knn_secondary_targets_data.empty()) {
        for (const auto& knn_data : knn_secondary_targets_data) {
            const Camera* secondary_camera = knn_data.first;
            const torch::Tensor& secondary_gt_image = knn_data.second; // Assumed [H,W,C] on device

            // Render this secondary view (simplified: actual render call needed)
            // gs::RenderOutput secondary_render_output = gs::rasterize(...);
            // For now, let's assume we have a placeholder or skip actual rendering for secondary targets
            // to avoid making this step too complex with re-rendering.
            // The paper says "Hessians and gradients of secondary targets are sparsely evaluated".
            // This implies a simplified rendering/evaluation for them.
            // Let's assume, for now, we reuse primary_render_output's structure but with secondary camera and GT.
            // This is a simplification! A proper implementation needs to render secondary views.

            // For simplicity, we'd need to re-rasterize for secondary targets.
            // This is beyond the scope of this single step.
            // So, this part remains conceptual:
            // LossDerivatives secondary_loss_derivs = compute_loss_derivatives_cuda(secondary_render_output.image, secondary_gt_image, ...);
            // PositionHessianOutput secondary_hess_output = compute_position_hessian_components_cuda(model_, visibility_mask_for_model, *secondary_camera, secondary_render_output, secondary_loss_derivs, num_visible_gaussians_in_model);
            // H_p_total_packed.add_(secondary_hess_output.H_p_packed);
            // g_p_total_visible.add_(secondary_hess_output.grad_p);
        }
    }

    // IV. Project Hessian and Gradient to 2D camera plane (U_k^T H U_k, U_k^T g)
    torch::Tensor grad_v_projected = torch::zeros({num_visible_gaussians_in_model, 2}, g_p_total_visible.options());
    torch::Tensor H_v_projected_packed = compute_projected_position_hessian_and_gradient(
        H_p_total_packed, g_p_total_visible, means_visible_from_model, primary_camera, grad_v_projected
    );

    // V & VI. Solve for Δv, re-project to Δp
    torch::Tensor delta_p = solve_and_project_position_updates(
        H_v_projected_packed, grad_v_projected, means_visible_from_model, primary_camera,
        options_.damping, options_.step_scale
    );

    // VII. Update model means
    model_.means().index_add_(0, visible_indices, delta_p);
}
