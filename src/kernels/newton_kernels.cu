// src/kernels/newton_kernels.cu
#include "kernels/newton_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/torch.h> // For AT_ASSERTM

// Basic CUDA utilities (normally in a separate header)
#define CUDA_CHECK(status) AT_ASSERTM(status == cudaSuccess, cudaGetErrorString(status))

constexpr int CUDA_NUM_THREADS = 256; // Default number of threads per block
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// --- KERNEL IMPLEMENTATIONS ---

// Kernel for dL/dc and d2L/dc2
__global__ void compute_loss_derivatives_kernel(
    const float* rendered_image, // [H, W, C]
    const float* gt_image,       // [H, W, C]
    float lambda_dssim,
    bool use_l2_loss_term,
    float* out_dL_dc,      // [H, W, C]
    float* out_d2L_dc2_diag, // [H, W, C]
    int H, int W, int C) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = H * W * C;

    if (idx >= total_elements) return;

    // TODO: Implement actual L1/L2 and SSIM derivative calculations here.
    // This is a placeholder. A full SSIM derivative is complex.
    // For L1:
    // float diff = rendered_image[idx] - gt_image[idx];
    // out_dL_dc[idx] = (diff > 0) ? 1.f : -1.f; // sign(diff)
    // if (abs(diff) < 1e-5) out_dL_dc[idx] = 0; // handle zero case
    // out_d2L_dc2_diag[idx] = 0.f; // d2(L1)/dc2 is 0 (problematic for paper's Hessian form)

    // For L2 (as in paper for the L2 term):
    float diff = rendered_image[idx] - gt_image[idx];
    if (use_l2_loss_term) {
        out_dL_dc[idx] = 2.f * diff;
        out_d2L_dc2_diag[idx] = 2.f;
    } else { // L1
        out_dL_dc[idx] = (diff > 1e-6f) ? 1.f : ((diff < -1e-6f) ? -1.f : 0.f);
        out_d2L_dc2_diag[idx] = 0.f; // For L1, 2nd derivative is zero
    }


    // The paper's SSIM derivatives (Eq 6-13) are per-pixel but depend on a window.
    // This kernel would need to access pixel neighborhoods for SSIM.
    // For simplicity in this step, we are only implementing the L1/L2 part.
    // A full SSIM derivative implementation is a separate, significant task.
    // The lambda_dssim would be used to weigh the SSIM components if they were calculated.
    // For example:
    // float l_term_dl_dc = out_dL_dc[idx];
    // float l_term_d2l_dc2 = out_d2L_dc2_diag[idx];
    // float ssim_term_dl_dc = ... calculate ...
    // float ssim_term_d2l_dc2 = ... calculate ...
    // out_dL_dc[idx] = (1.f - lambda_dssim) * l_term_dl_dc + lambda_dssim * ssim_term_dl_dc;
    // out_d2L_dc2_diag[idx] = (1.f - lambda_dssim) * l_term_d2l_dc2 + lambda_dssim * ssim_term_d2l_dc2;
}


// Kernel for position Hessian components
// This is a very complex kernel. The sketch below is highly simplified.
// It needs to implement parts of rasterization forward and then derivatives.
__global__ void compute_position_hessian_components_kernel(
    int H_img, int W_img, int C_img,
    int P_total,
    const float* means_3d_all, const float* scales_all, const float* rotations_all,
    const float* opacities_all, const float* shs_all, int sh_degree, int sh_coeffs_dim,
    const float* view_matrix, const float* projection_matrix_for_jacobian, const float* cam_pos_world,
    const float* means_2d_render, const float* depths_render, const float* radii_render,
    const int* visibility_indices_in_render_output, int P_render,
    const bool* visibility_mask_for_model,
    const float* dL_dc_pixelwise, const float* d2L_dc2_diag_pixelwise,
    int num_output_gaussians,
    float* H_p_output_packed, float* grad_p_output,
    // Helper: map original P_total index to dense output index (0 to num_output_gaussians-1)
    // This map should be precomputed on CPU and passed if outputs are dense.
    // If visibility_indices_in_render_output is not null, it might serve a similar purpose
    // for mapping render output data.
    const int* output_index_map // [P_total], value is output slot or -1 if not visible
) {
    int p_idx_total = blockIdx.x * blockDim.x + threadIdx.x; // Iterate over all Gaussians in model

    if (p_idx_total >= P_total) return;
    if (!visibility_mask_for_model[p_idx_total]) return;

    int output_idx = output_index_map ? output_index_map[p_idx_total] : -1;
    if (output_idx == -1) return; // Should not happen if visibility_mask_for_model[p_idx_total] is true and map is correct

    // For each visible Gaussian p_idx_total:
    // 1. Get its parameters (means_3d_all[p_idx_total*3], etc.)
    // 2. Compute its influence on pixels (coverage, color contribution). This is part of rasterization.
    //    This step is very complex. It involves projecting the Gaussian, calculating its 2D covariance,
    //    determining which pixels it covers.
    //
    // 3. For each covered pixel (px, py):
    //    a. Get dL/dc(px,py) and d2L/dc2(px,py) from inputs.
    //    b. Compute Jacobian J_c(px,py) = 𝜕c(px,py)/𝜕p_k (how this Gaussian's position change affects this pixel color).
    //       This involves derivatives of projection, SH evaluation, Gaussian PDF, alpha blending. (Eq 16 from paper)
    //    c. Compute Hessian H_c_y(px,py) = 𝜕²c(px,py)/𝜕p_k² (second derivative). (Eq 17 from paper)
    //
    //    d. Accumulate to H_p_k and g_p_k for this Gaussian p_idx_total:
    //       g_p_k += J_c(px,py)^T * dL/dc(px,py)
    //       H_p_k += J_c(px,py)^T * d2L/dc2(px,py) * J_c(px,py) + dL/dc(px,py) * H_c_y(px,py)
    //
    // 4. Store H_p_k (packed) and g_p_k into H_p_output_packed[output_idx*6] and grad_p_output[output_idx*3].

    // --- Placeholder ---
    // This is extremely simplified. A real implementation is hundreds of lines of CUDA.
    // Initialize H_p_k and g_p_k to zero
    float g_p_k[3] = {0,0,0};
    float H_p_k_symm[6] = {0,0,0,0,0,0}; // H00, H01, H02, H11, H12, H22

    // Dummy values for demonstration
    g_p_k[0] = 1.0f * p_idx_total; g_p_k[1] = 0.5f * p_idx_total; g_p_k[2] = 0.1f * p_idx_total;
    H_p_k_symm[0] = 1.0f; // H00
    H_p_k_symm[3] = 1.0f; // H11
    H_p_k_symm[5] = 1.0f; // H22

    for(int i=0; i<3; ++i) grad_p_output[output_idx * 3 + i] = g_p_k[i];
    for(int i=0; i<6; ++i) H_p_output_packed[output_idx * 6 + i] = H_p_k_symm[i];
}


// Kernel for projecting Hessian and Gradient
__global__ void project_position_hessian_gradient_kernel(
    int num_visible_gaussians,
    const float* H_p_packed_input, // [N_vis, 6] (Hpxx, Hpxy, Hpxz, Hpyy, Hpyz, Hpzz)
    const float* grad_p_input,     // [N_vis, 3]
    const float* means_3d_visible, // [N_vis, 3]
    const float* view_matrix,      // [16] (col-major or row-major assumed by caller)
    const float* cam_pos_world,    // [3]
    float* out_H_v_packed,         // [N_vis, 3] (Hvxx, Hvxy, Hvyy)
    float* out_grad_v) {           // [N_vis, 2]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_visible_gaussians) return;

    // 1. Calculate r_k = p_k - cam_pos_world (view vector from camera to point)
    //    It's often more convenient to use the camera's Z-axis (view direction) and up/right vectors.
    //    The paper's U_k = [u_x, u_y] forms a 2D basis perpendicular to r_k.
    //    Let r_k = means_3d_visible[idx*3+c] - cam_pos_world[c]
    //    This r_k is pointing from camera to Gaussian.
    //    The paper mentions "camera's look at r_k". This usually means r_k is the direction vector.
    //    Let's assume view_matrix gives camera orientation. view_matrix[2], view_matrix[6], view_matrix[10] is cam Z axis (if row-major).
    //    Let world_z_vec = {view_matrix[2], view_matrix[6], view_matrix[10]} (camera's forward vector)
    //    Let world_y_vec = {view_matrix[1], view_matrix[5], view_matrix[9]} (camera's up vector)
    //    Let world_x_vec = {view_matrix[0], view_matrix[4], view_matrix[8]} (camera's right vector)

    // Simplified U_k: use camera's X and Y axes in world space as u_x, u_y.
    // This assumes planar adjustment aligned with camera's own axes.
    // Paper Eq 14 is more complex: u_y = (r_k outer_prod r_k)[0,1,0]^T / norm(...)
    // This implies r_k is used to define the plane.
    // For now, let u_x = camera right, u_y = camera up. This is a common simplification for screen-space operations.
    float ux[3] = {view_matrix[0], view_matrix[4], view_matrix[8]}; // Camera Right
    float uy[3] = {view_matrix[1], view_matrix[5], view_matrix[9]}; // Camera Up

    // Project gradient: g_v = U^T g_p
    // g_v[0] = ux . grad_p_input[idx*3+c]
    // g_v[1] = uy . grad_p_input[idx*3+c]
    out_grad_v[idx*2 + 0] = ux[0]*grad_p_input[idx*3+0] + ux[1]*grad_p_input[idx*3+1] + ux[2]*grad_p_input[idx*3+2];
    out_grad_v[idx*2 + 1] = uy[0]*grad_p_input[idx*3+0] + uy[1]*grad_p_input[idx*3+1] + uy[2]*grad_p_input[idx*3+2];

    // Project Hessian: H_v = U^T H_p U
    // H_p matrix from packed:
    // [ H00 H01 H02 ]
    // [ H01 H11 H12 ]
    // [ H02 H12 H22 ]
    // H_p_packed_input = [H00, H01, H02, H11, H12, H22]
    const float* Hp = &H_p_packed_input[idx*6];
    float Hpu_x[3]; // H_p * u_x
    Hpu_x[0] = Hp[0]*ux[0] + Hp[1]*ux[1] + Hp[2]*ux[2];
    Hpu_x[1] = Hp[1]*ux[0] + Hp[3]*ux[1] + Hp[4]*ux[2];
    Hpu_x[2] = Hp[2]*ux[0] + Hp[4]*ux[1] + Hp[5]*ux[2];

    float Hpu_y[3]; // H_p * u_y
    Hpu_y[0] = Hp[0]*uy[0] + Hp[1]*uy[1] + Hp[2]*uy[2];
    Hpu_y[1] = Hp[1]*uy[0] + Hp[3]*uy[1] + Hp[4]*uy[2];
    Hpu_y[2] = Hp[2]*uy[0] + Hp[4]*uy[1] + Hp[5]*uy[2];

    // H_v elements:
    // Hv_xx = u_x^T H_p u_x
    out_H_v_packed[idx*3 + 0] = ux[0]*Hpu_x[0] + ux[1]*Hpu_x[1] + ux[2]*Hpu_x[2];
    // Hv_xy = u_x^T H_p u_y
    out_H_v_packed[idx*3 + 1] = ux[0]*Hpu_y[0] + ux[1]*Hpu_y[1] + ux[2]*Hpu_y[2];
    // Hv_yy = u_y^T H_p u_y
    out_H_v_packed[idx*3 + 2] = uy[0]*Hpu_y[0] + uy[1]*Hpu_y[1] + uy[2]*Hpu_y[2];
}

// Kernel for batch 2x2 solve
__global__ void batch_solve_2x2_system_kernel(
    int num_systems,
    const float* H_v_packed, // [N, 3] (H00, H01, H11)
    const float* g_v,        // [N, 2] (g0, g1)
    float damping,
    float step_scale,
    float* out_delta_v) {    // [N, 2]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_systems) return;

    float H00 = H_v_packed[idx*3 + 0];
    float H01 = H_v_packed[idx*3 + 1];
    float H11 = H_v_packed[idx*3 + 2];

    float g0 = g_v[idx*2 + 0];
    float g1 = g_v[idx*2 + 1];

    // Add damping to diagonal
    H00 += damping;
    H11 += damping;

    float det = H00 * H11 - H01 * H01;

    // If det is too small, effectively no update or use gradient descent step
    if (abs(det) < 1e-8f) {
        out_delta_v[idx*2 + 0] = -step_scale * g0 / (H00 + 1e-6f); // Simplified fallback
        out_delta_v[idx*2 + 1] = -step_scale * g1 / (H11 + 1e-6f);
        return;
    }

    float inv_det = 1.f / det;

    // delta_v = - H_inv * g
    // H_inv = inv_det * [H11, -H01; -H01, H00]
    out_delta_v[idx*2 + 0] = -step_scale * inv_det * (H11 * g0 - H01 * g1);
    out_delta_v[idx*2 + 1] = -step_scale * inv_det * (-H01 * g0 + H00 * g1);
}

// Kernel for re-projecting delta_v to delta_p
__global__ void project_update_to_3d_kernel(
    int num_updates,
    const float* delta_v,          // [N, 2] (dvx, dvy)
    const float* means_3d_visible, // [N, 3] (Not strictly needed if U_k doesn't depend on p_k itself, but paper's U_k does via r_k)
    const float* view_matrix,      // [16]
    const float* cam_pos_world,    // [3]
    float* out_delta_p) {          // [N, 3]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_updates) return;

    // Using the same simplified U_k = [cam_right, cam_up] as in projection kernel
    float ux[3] = {view_matrix[0], view_matrix[4], view_matrix[8]}; // Camera Right
    float uy[3] = {view_matrix[1], view_matrix[5], view_matrix[9]}; // Camera Up

    float dvx = delta_v[idx*2 + 0];
    float dvy = delta_v[idx*2 + 1];

    // delta_p = U_k * delta_v = u_x * dvx + u_y * dvy
    out_delta_p[idx*3 + 0] = ux[0] * dvx + uy[0] * dvy;
    out_delta_p[idx*3 + 1] = ux[1] * dvx + uy[1] * dvy;
    out_delta_p[idx*3 + 2] = ux[2] * dvx + uy[2] * dvy;
}


// --- LAUNCHER FUNCTIONS ---

void NewtonKernels::compute_loss_derivatives_kernel_launcher(
    const torch::Tensor& rendered_image_tensor,
    const torch::Tensor& gt_image_tensor,
    float lambda_dssim,
    bool use_l2_loss_term,
    torch::Tensor& out_dL_dc_tensor,
    torch::Tensor& out_d2L_dc2_diag_tensor) {

    int H = rendered_image_tensor.size(0);
    int W = rendered_image_tensor.size(1);
    int C = rendered_image_tensor.size(2);
    int total_elements = H * W * C;

    const float* rendered_image_ptr = gs::torch_utils::get_const_data_ptr<float>(rendered_image_tensor);
    const float* gt_image_ptr = gs::torch_utils::get_const_data_ptr<float>(gt_image_tensor);
    float* out_dL_dc_ptr = gs::torch_utils::get_data_ptr<float>(out_dL_dc_tensor);
    float* out_d2L_dc2_diag_ptr = gs::torch_utils::get_data_ptr<float>(out_d2L_dc2_diag_tensor);

    compute_loss_derivatives_kernel<<<GET_BLOCKS(total_elements), CUDA_NUM_THREADS>>>(
        rendered_image_ptr, gt_image_ptr, lambda_dssim, use_l2_loss_term,
        out_dL_dc_ptr, out_d2L_dc2_diag_ptr, H, W, C
    );
    CUDA_CHECK(cudaGetLastError());
}


void NewtonKernels::compute_position_hessian_components_kernel_launcher(
    int H_img, int W_img, int C_img,
    int P_total,
    const float* means_3d_all, const float* scales_all, const float* rotations_all,
    const float* opacities_all, const float* shs_all, int sh_degree, int sh_coeffs_dim,
    const float* view_matrix, const float* projection_matrix_for_jacobian, const float* cam_pos_world,
    const float* means_2d_render, const float* depths_render, const float* radii_render,
    const int* visibility_indices_in_render_output, int P_render,
    const bool* visibility_mask_for_model,
    const float* dL_dc_pixelwise, const float* d2L_dc2_diag_pixelwise,
    int num_output_gaussians,
    float* H_p_output_packed, float* grad_p_output
) {
    // Precompute output_index_map on CPU/GPU
    // output_index_map: array of size P_total. output_index_map[i] is the dense output index for Gaussian i, or -1.
    // This map is crucial. For now, assuming it's passed or P_total is small enough for simple handling.
    // This is a placeholder for the actual kernel call, which needs the map.
    // For now, the kernel is simplified and assumes P_total is the number of output gaussians if output_index_map is null.
    // This needs to be fixed for a real scenario.

    // Construct the output_index_map (example of how it might be done on CPU and passed)
    std::vector<int> output_index_map_cpu(P_total);
    int current_out_idx = 0;
    for(int i=0; i<P_total; ++i) {
        if(visibility_mask_for_model[i]) {
            output_index_map_cpu[i] = current_out_idx++;
        } else {
            output_index_map_cpu[i] = -1;
        }
    }
    // AT_ASSERTM(current_out_idx == num_output_gaussians, "Mismatch in visible count for output_index_map");

    torch::Tensor output_index_map_tensor = torch::tensor(output_index_map_cpu, torch::kInt).to(torch::kCUDA);
    const int* output_index_map_gpu = gs::torch_utils::get_const_data_ptr<int>(output_index_map_tensor);


    compute_position_hessian_components_kernel<<<GET_BLOCKS(P_total), CUDA_NUM_THREADS>>>(
        H_img, W_img, C_img, P_total, means_3d_all, scales_all, rotations_all, opacities_all,
        shs_all, sh_degree, sh_coeffs_dim, view_matrix, projection_matrix_for_jacobian, cam_pos_world,
        means_2d_render, depths_render, radii_render, visibility_indices_in_render_output, P_render,
        visibility_mask_for_model, dL_dc_pixelwise, d2L_dc2_diag_pixelwise,
        num_output_gaussians, H_p_output_packed, grad_p_output,
        output_index_map_gpu // Pass the map
    );
    CUDA_CHECK(cudaGetLastError());
}

void NewtonKernels::project_position_hessian_gradient_kernel_launcher(
    int num_visible_gaussians,
    const float* H_p_packed_input, const float* grad_p_input,
    const float* means_3d_visible, const float* view_matrix,
    const float* cam_pos_world,
    float* out_H_v_packed, float* out_grad_v ) {

    project_position_hessian_gradient_kernel<<<GET_BLOCKS(num_visible_gaussians), CUDA_NUM_THREADS>>>(
        num_visible_gaussians, H_p_packed_input, grad_p_input, means_3d_visible,
        view_matrix, cam_pos_world, out_H_v_packed, out_grad_v
    );
    CUDA_CHECK(cudaGetLastError());
}

void NewtonKernels::batch_solve_2x2_system_kernel_launcher(
    int num_systems,
    const float* H_v_packed, const float* g_v, float damping, float step_scale,
    float* out_delta_v ) {

    batch_solve_2x2_system_kernel<<<GET_BLOCKS(num_systems), CUDA_NUM_THREADS>>>(
        num_systems, H_v_packed, g_v, damping, step_scale, out_delta_v
    );
    CUDA_CHECK(cudaGetLastError());
}

void NewtonKernels::project_update_to_3d_kernel_launcher(
    int num_updates,
    const float* delta_v, const float* means_3d_visible,
    const float* view_matrix, const float* cam_pos_world,
    float* out_delta_p ) {

    project_update_to_3d_kernel<<<GET_BLOCKS(num_updates), CUDA_NUM_THREADS>>>(
        num_updates, delta_v, means_3d_visible, view_matrix, cam_pos_world, out_delta_p
    );
    CUDA_CHECK(cudaGetLastError());
}

// Make sure torch_utils.hpp has these definitions or similar:
// namespace gs { namespace torch_utils {
// template <typename T>
// inline const T* get_const_data_ptr(const torch::Tensor& tensor) {
//     TORCH_CHECK(tensor.is_cuda(), "Tensor must be CUDA tensor");
//     TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
//     return tensor.data_ptr<T>();
// }
// template <typename T>
// inline T* get_data_ptr(torch::Tensor& tensor) {
//     TORCH_CHECK(tensor.is_cuda(), "Tensor must be CUDA tensor");
//     TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
//     return tensor.data_ptr<T>();
// }
// }}
