/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <functional>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(float2* xy_d,
    	float *depths_d,
		int *radii_d,
		int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* dc,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const uint32_t* per_tile_bucket_offset, uint32_t* bucket_to_tile,
		float* sampled_T, float* sampled_ar,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		uint32_t* max_contrib,
		float* pixel_colors,
		const float* bg_color,
		float* out_color,
		int* img_contribs,
		int* img_contrib_offsets,
		char* img_contrib_scan,
		size_t scan_size,
		std::function<int* (size_t)> listBuffer,
		std::function<float* (size_t)> listBufferRender,
		std::function<float* (size_t)> listBufferDistance,
		float* pixel_weights,
		float* accum_weights,
		int* reverse_count,
		float* blend_weights,
		float* dist_accum);
}


#endif