// kernels/newton_kernels.cuh
#pragma once
#include <torch/torch.h>
#include "core/torch_utils.hpp" // For gs::torch_utils like get_data_ptr

namespace NewtonKernels {

// Launcher for computing dL/dc and d2L/dc2
void compute_loss_derivatives_kernel_launcher(
    const torch::Tensor& rendered_image_tensor, // [H, W, C]
    const torch::Tensor& gt_image_tensor,       // [H, W, C]
    float lambda_dssim,
    bool use_l2_loss_term, // If true, L2+SSIM for L, else L1+SSIM
    torch::Tensor& out_dL_dc_tensor,      // [H, W, C]
    torch::Tensor& out_d2L_dc2_diag_tensor // [H, W, C] (diagonal elements)
);

// Launcher for computing Hessian components for position
// H_p = J_c^T * H_L_c * J_c + G_L_c * H_c_y
// g_p = J_c^T * G_L_c
// J_c = 𝜕𝒄 / 𝜕𝒑𝑘 (Jacobian of final pixel color w.r.t. p_k)
// H_L_c = 𝜕²L/𝜕c² (Hessian of loss w.r.t. final pixel color)
// G_L_c = 𝜕L/𝜕c (Gradient of loss w.r.t. final pixel color)
// H_c_y = 𝜕²𝒄/𝜕p² (Hessian of final pixel color w.r.t. p_k)
void compute_position_hessian_components_kernel_launcher(
    // Image dimensions
    int H, int W, int C_img, // C_img is number of channels in image (e.g. 3 for RGB)
    // Gaussian properties (all Gaussians in the model)
    int P_total,
    const float* means_3d_all,
    const float* scales_all,
    const float* rotations_all,
    const float* opacities_all,
    const float* shs_all,
    int sh_degree,
    int sh_coeffs_dim, // total dimension of SH coeffs per channel (e.g., (sh_degree+1)^2)
    // Camera properties
    const float* view_matrix,
    const float* projection_matrix_for_jacobian, // Typically K matrix [3,3] or [4,4]
    const float* cam_pos_world,
    // Data from RenderOutput (for Gaussians processed by rasterizer, potentially culled)
    const float* means_2d_render,  // [P_render, 2]
    const float* depths_render,   // [P_render]
    const float* radii_render,    // [P_render]
    const int* visibility_indices_in_render_output, // [P_render] maps P_render index to P_total index. If nullptr, means_2d_render is for P_total.
    int P_render, // Number of Gaussians in means_2d_render etc.
    // Visibility mask for *all* Gaussians in the model [P_total]. True if Gaussian k is visible on screen.
    const bool* visibility_mask_for_model,
    // Loss derivatives (pixel-wise)
    const float* dL_dc_pixelwise,          // [H, W, C_img]
    const float* d2L_dc2_diag_pixelwise,   // [H, W, C_img]
    // Output arrays are for Gaussians where visibility_mask_for_model is true.
    // num_output_gaussians is the count of true in visibility_mask_for_model.
    int num_output_gaussians,
    // Output arrays (dense, for visible Gaussians from model)
    float* H_p_output_packed, // [num_output_gaussians, 6] (symmetric 3x3)
    float* grad_p_output      // [num_output_gaussians, 3]
);


// Launcher for projecting Hessian and Gradient to camera plane
void project_position_hessian_gradient_kernel_launcher(
    int num_visible_gaussians,
    const float* H_p_packed_input,
    const float* grad_p_input,
    const float* means_3d_visible,
    const float* view_matrix,
    const float* cam_pos_world,
    float* out_H_v_packed,
    float* out_grad_v
);

// Launcher for solving batch 2x2 linear systems H_v * delta_v = -g_v
void batch_solve_2x2_system_kernel_launcher(
    int num_systems,
    const float* H_v_packed,
    const float* g_v,
    float damping,
    float step_scale, // Applied as: delta_v = -step_scale * H_inv * g
    float* out_delta_v
);

// Launcher for re-projecting delta_v to 3D delta_p = U_k * delta_v
void project_update_to_3d_kernel_launcher(
    int num_updates,
    const float* delta_v,
    const float* means_3d_visible,
    const float* view_matrix,
    const float* cam_pos_world,
    float* out_delta_p
);

} // namespace NewtonKernels
