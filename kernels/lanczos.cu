/*
Copyright (c) 2025 Youyu Chen
SPDX-License-Identifier: MIT
*/
#include <math.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "kernels/lanczos.cuh"

namespace cg = cooperative_groups;

namespace CUDA_LANCZOS {

__device__ float sinc(const float x) {
	if (fabs(x) < 1e-12) return 1.0f;
	return sin(M_PI * x) / (M_PI * x);
}

__device__ float lanczos_kernel(const float x, const float a) {
	if (x <= -a || x >= a) return 0.0f;
	return sinc(x) * sinc(x / a);
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
PreComputeCoef(
	const int input_size, 
	const int output_size, 
	const int kernel_size, 
	float* __restrict__ kernel_values
) {
	const auto block = cg::this_thread_block();
	const uint32_t thread_idx = block.thread_index().x;
	const uint32_t output_idx = block.group_index().x * BLOCK_X * BLOCK_Y + thread_idx;
	const float output_ax = (float)output_idx;
	const float scale = 1.0f * input_size / output_size;
	const bool inside = output_idx < output_size;

	if (!inside) return;

	const float center = (output_ax + 0.5f) * scale;
	const int2 box = {
		max((int)(center - kernel_size * scale + 0.5f), 0), 
		min((int)(center + kernel_size * scale + 0.5f), input_size)
	};

	const uint32_t offset = output_idx * (uint32_t)(kernel_size * scale * 2 + 1 + 0.5f);
	float norm = 0.0f;
	for (int i = box.x; i < box.y; i++) {
		float value = lanczos_kernel((i + 0.5f - center) / scale, kernel_size);
		kernel_values[offset + i - box.x] = value;
		norm += value;
	}
	for (int i = box.x; i < box.y; i++) {
		kernel_values[offset + i - box.x] /= norm;
	}
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
LanczosResampleCUDA(
	const int input_h, const int input_w, 
	const int output_h, const int output_w, 
	const int kernel_size, 
	const float* __restrict__ pre_coef_x, 
	const float* __restrict__ pre_coef_y, 
	const float* __restrict__ input, 
	float* __restrict__ output
) {
	const auto block = cg::this_thread_block();
	const uint32_t thread_idx_x = block.thread_index().x;
	const uint32_t thread_idx_y = block.thread_index().y;
	const uint2 pix = {
		block.group_index().x * BLOCK_X + thread_idx_x, 
		block.group_index().y * BLOCK_Y + thread_idx_y
	};
	const uint32_t pix_id = output_w * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	float scale_h = 1.0f * input_h / output_h, scale_w = 1.0f * input_w / output_w;

	const bool inside = (pix.x < output_w && pix.y < output_h);

	if (!inside) return;

	const float2 center = { (pixf.x + 0.5f) * scale_w, (pixf.y + 0.5f) * scale_h };

	const int2 LU = {
		max((int)(center.x - kernel_size * scale_w + 0.5f), 0), 
		max((int)(center.y - kernel_size * scale_h + 0.5f), 0)
	};
	const int2 RD = {
		min((int)(center.x + kernel_size * scale_w + 0.5f), input_w), 
		min((int)(center.y + kernel_size * scale_h + 0.5f), input_h)
	};

	uint32_t coef_offset_step_y = (uint32_t)(kernel_size * scale_h * 2 + 1 + 0.5f);
	uint32_t coef_offset_step_x = (uint32_t)(kernel_size * scale_w * 2 + 1 + 0.5f);
	for (int y = LU.y; y < RD.y; y++) {
		float kernel_value_y = pre_coef_y[pix.y * coef_offset_step_y + y - LU.y];
		for (int x = LU.x; x < RD.x; x++) {
			uint32_t input_pix_id = input_w * y + x;
			float kernel_value_x = pre_coef_x[pix.x * coef_offset_step_x + x - LU.x];
			float kernel_value = kernel_value_y * kernel_value_x;
			for (int ch = 0; ch < CHANNELS; ch++)
				output[pix_id * 3 + ch] += input[input_pix_id * 3 + ch] * kernel_value;
		}
	}
}

}

torch::Tensor LanczosResampling(
	const torch::Tensor &input, 
	const int output_h, 
	const int output_w, 
	const int kernel_size
) {
	const int input_h = input.size(0), input_w = input.size(1);
	uint32_t channels = input.size(2);
	torch::Tensor output = torch::zeros({output_h, output_w, channels}, input.options());

	const uint32_t offset_step_x = (uint32_t)(kernel_size * (1.0 * input_w / output_w) * 2 + 1 + 0.5f);
	const uint32_t offset_step_y = (uint32_t)(kernel_size * (1.0 * input_h / output_h) * 2 + 1 + 0.5f);
	float* coef_x;
	float* coef_y;
	cudaMalloc(&coef_x, sizeof(float) * output_w * offset_step_x);
	cudaMalloc(&coef_y, sizeof(float) * output_h * offset_step_y);

	CUDA_LANCZOS::PreComputeCoef<<<(output_w + BLOCK_X * BLOCK_Y - 1) / (BLOCK_X * BLOCK_Y), BLOCK_X * BLOCK_Y>>>(
		input_w, output_w, kernel_size, coef_x);
	CUDA_LANCZOS::PreComputeCoef<<<(output_h + BLOCK_X * BLOCK_Y - 1) / (BLOCK_X * BLOCK_Y), BLOCK_X * BLOCK_Y>>>(
		input_h, output_h, kernel_size, coef_y);
	
	const dim3 tile_grid((output_w + BLOCK_X - 1) / BLOCK_X, (output_h + BLOCK_Y - 1) / BLOCK_Y);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	CUDA_LANCZOS::LanczosResampleCUDA<NUM_CHANNELS><<<tile_grid, block>>>(
		input_h, input_w, 
		output_h, output_w, 
		kernel_size, 
		coef_x, 
		coef_y, 
		input.contiguous().data_ptr<float>(), 
		output.contiguous().data_ptr<float>()
	);

	cudaFree(coef_x);
	cudaFree(coef_y);

	return output;
}