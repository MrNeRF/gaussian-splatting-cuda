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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

__device__ inline float evaluate_opacity_factor(const float dx, const float dy, const float4 co) {
	return 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
}

template<uint32_t PATCH_WIDTH, uint32_t PATCH_HEIGHT>
__device__ inline float max_contrib_power_rect_gaussian_float(
	const float4 co, 
	const float2 mean, 
	const glm::vec2 rect_min,
	const glm::vec2 rect_max,
	glm::vec2& max_pos)
{
	const float x_min_diff = rect_min.x - mean.x;
	const float x_left = x_min_diff > 0.0f;
	// const float x_left = mean.x < rect_min.x;
	const float not_in_x_range = x_left + (mean.x > rect_max.x);

	const float y_min_diff = rect_min.y - mean.y;
	const float y_above =  y_min_diff > 0.0f;
	// const float y_above = mean.y < rect_min.y;
	const float not_in_y_range = y_above + (mean.y > rect_max.y);

	max_pos = {mean.x, mean.y};
	float max_contrib_power = 0.0f;

	if ((not_in_y_range + not_in_x_range) > 0.0f)
	{
		const float px = x_left * rect_min.x + (1.0f - x_left) * rect_max.x;
		const float py = y_above * rect_min.y + (1.0f - y_above) * rect_max.y;

		const float dx = copysign(float(PATCH_WIDTH), x_min_diff);
		const float dy = copysign(float(PATCH_HEIGHT), y_min_diff);

		const float diffx = mean.x - px;
		const float diffy = mean.y - py;

		const float rcp_dxdxcox = __frcp_rn(PATCH_WIDTH * PATCH_WIDTH * co.x); // = 1.0 / (dx*dx*co.x)
		const float rcp_dydycoz = __frcp_rn(PATCH_HEIGHT * PATCH_HEIGHT * co.z); // = 1.0 / (dy*dy*co.z)

		const float tx = not_in_y_range * __saturatef((dx * co.x * diffx + dx * co.y * diffy) * rcp_dxdxcox);
		const float ty = not_in_x_range * __saturatef((dy * co.y * diffx + dy * co.z * diffy) * rcp_dydycoz);
		max_pos = {px + tx * dx, py + ty * dy};
		
		const float2 max_pos_diff = {mean.x - max_pos.x, mean.y - max_pos.y};
		max_contrib_power = evaluate_opacity_factor(max_pos_diff.x, max_pos_diff.y, co);
	}

	return max_contrib_power;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

__device__ __forceinline__
int2 getExtRect_fast(const float4& cov) {
    // log2(opacity)
    float l2;
    asm volatile ("lg2.approx.f32 %0, %1;" : "=f"(l2) : "f"(cov.w));

    // power = ln(2) * (8 + log2(opacity))
    const float ln2   = 0.69314718055f;
    const float ln2x8 = 5.5451774444f;          // ln2 * 8
    float power = __fmaf_rn(l2, ln2, ln2x8);

    // sx = cov.x * power, sy = cov.z * power
    float det = cov.x * cov.z - cov.y * cov.y;
    if (det <= 0.0f) return {0, 0};
    float sx = cov.z * power / det;
    float sy = cov.x * power / det;

    // sqrt.approx
    float rx, ry;
    asm volatile ("sqrt.approx.f32 %0, %1;" : "=f"(rx) : "f"(sx));
    asm volatile ("sqrt.approx.f32 %0, %1;" : "=f"(ry) : "f"(sy));

    // width/height = ceil(sqrt2 * r + 1); use fma + trunc toward zero
    // const float sqrt2 = 1.41421356f;
    // int w = __float2int_rz(__fmaf_rn(rx, sqrt2, 1.0f));
    // int h = __float2int_rz(__fmaf_rn(ry, sqrt2, 1.0f));

    return {rx + 1, ry + 1};
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float4* __restrict__ conic_opacity,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid,
	int2* rects)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		const uint32_t offset_to = offsets[idx];
		uint2 rect_min, rect_max;

		// if(rects == nullptr)
			// getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
		// else
			// getRect(points_xy[idx], rects[idx], rect_min, rect_max, grid);

		const float4 co = conic_opacity[idx];
		getRect(points_xy[idx], getExtRect_fast(co), rect_min, rect_max, grid);

		const float2 xy = points_xy[idx];
		// const float4 co = conic_opacity[idx];
		const float opacity_threshold = 1.0f / 255.0f;
		const float opacity_factor_threshold = logf(co.w / opacity_threshold);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				const glm::vec2 tile_min(x * BLOCK_X, y * BLOCK_Y);
				const glm::vec2 tile_max((x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1);

				glm::vec2 max_pos;
				float max_opac_factor = 0.0f;
				max_opac_factor = max_contrib_power_rect_gaussian_float<BLOCK_X-1, BLOCK_Y-1>(co, xy, tile_min, tile_max, max_pos);
				
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				if (max_opac_factor <= opacity_factor_threshold) {
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}

		for (; off < offset_to; ++off) {
			uint64_t key = (uint32_t) -1;
			key <<= 32;
			const float depth = FLT_MAX;
			key |= *((uint32_t*)&depth);
			gaussian_values_unsorted[off] = static_cast<uint32_t>(-1);
			gaussian_keys_unsorted[off] = key;
		}
	}
}

// extern "C" __global__
// void identifyTileRanges(
// 	int L,
// 	const uint64_t* __restrict__ point_list_keys,
//     uint2* __restrict__ ranges,
//     uint32_t   invalid = 0xFFFFFFFFF
// ) {

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= L) return;

//     uint32_t curr = point_list_keys[idx] >> 32;
//     bool curr_valid = (curr != invalid);

//     if (idx == 0) {
//         if (curr_valid) ranges[curr].x = 0;
//     } else {
//         uint32_t prev = point_list_keys[idx-1] >> 32;
//         if (curr != prev) {
//             if (prev != invalid) ranges[prev].y = idx;
//             if (curr_valid) ranges[curr].x = idx;
//         }
//     }

//     if (idx == L-1 && curr_valid) ranges[curr].y = L;

// }

__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	bool valid_tile = currtile != (uint32_t) -1;

	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			if (valid_tile) 
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1 && valid_tile)
		ranges[currtile].y = L;
}


// for each tile, see how many buckets/warps are needed to store the state
__global__ void perTileBucketCount(int T, uint2* ranges, uint32_t* bucketCount) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= T)
		return;
	
	uint2 range = ranges[idx];
	int num_splats = range.y - range.x;
	int num_buckets = (num_splats + 31) / 32;
	bucketCount[idx] = (uint32_t) num_buckets;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	int* dummy;
	int* wummy;
	cub::DeviceScan::InclusiveSum(nullptr, img.scan_size, dummy, wummy, N);
	obtain(chunk, img.contrib_scan, img.scan_size, 128);

	obtain(chunk, img.max_contrib, N, 128);
	obtain(chunk, img.pixel_colors, N * NUM_CHAFFELS, 128);
	obtain(chunk, img.bucket_count, N, 128);
	obtain(chunk, img.bucket_offsets, N, 128);
	cub::DeviceScan::InclusiveSum(nullptr, img.bucket_count_scan_size, img.bucket_count, img.bucket_count, N);
	obtain(chunk, img.bucket_count_scanning_space, img.bucket_count_scan_size, 128);

	return img;
}

CudaRasterizer::SampleState CudaRasterizer::SampleState::fromChunk(char *& chunk, size_t C) {
	SampleState sample;
	obtain(chunk, sample.bucket_to_tile, C * BLOCK_SIZE, 128);
	obtain(chunk, sample.T, C * BLOCK_SIZE, 128);
	obtain(chunk, sample.ar, NUM_CHAFFELS * C * BLOCK_SIZE, 128);
	return sample;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

__global__ void zero(int N, int* space)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= N)
		return;
	space[idx] = 0;
}

__global__ void set(int N, uint32_t* where, int* space)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= N)
		return;

	int off = (idx == 0) ? 0 : where[idx-1];

	space[off] = 1;
}


// Forward rendering procedure for differentiable rasterization
// of Gaussians.
std::tuple<int,int> CudaRasterizer::Rasterizer::forward(
	float2* xy_d,
    float *depths_d,
	int *radii_d,
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	std::function<char* (size_t)> sampleBuffer,
	std::function<int* (size_t)> listBuffer,
	std::function<float* (size_t)> listBufferRender,
	std::function<float* (size_t)> listBufferDistance,
	int* contribCountBuffer, int* contribOffsetBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* dc,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	bool debug,
	float* pixel_weights,
	float* accum_weights,
	int* reverse_count,
	float* blend_weights,
	float* dist_accum)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHAFFELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		xy_d, depths_d, radii_d,
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		dc,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid,
		nullptr);
	CHECK_CUDA(, debug);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug);

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug);

 	// bucket count
	int num_tiles = tile_grid.x * tile_grid.y;
	perTileBucketCount<<<(num_tiles + 255) / 256, 256>>>(num_tiles, imgState.ranges, imgState.bucket_count);
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(imgState.bucket_count_scanning_space, imgState.bucket_count_scan_size, imgState.bucket_count, imgState.bucket_offsets, num_tiles), debug);
	unsigned int bucket_sum;
	CHECK_CUDA(cudaMemcpy(&bucket_sum, imgState.bucket_offsets + num_tiles - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost), debug);
	// create a state to store. size is number is the total number of buckets * block_size
	size_t sample_chunk_size = required<SampleState>(bucket_sum);
	char* sample_chunkptr = sampleBuffer(sample_chunk_size);
	SampleState sampleState = SampleState::fromChunk(sample_chunkptr, bucket_sum);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		imgState.bucket_offsets, sampleState.bucket_to_tile,
		sampleState.T, sampleState.ar,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.max_contrib,
		imgState.pixel_colors,
		background,
		out_color,
		contribCountBuffer,
		contribOffsetBuffer,
		imgState.contrib_scan,
		imgState.scan_size,
		listBuffer, listBufferRender, listBufferDistance,
		pixel_weights,
		accum_weights,
		reverse_count,
		blend_weights,
		dist_accum), debug);

	return std::make_tuple(num_rendered, bucket_sum);
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R, int B,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* dc,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	char* sample_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_ddc,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);
	SampleState sampleState = SampleState::fromChunk(sample_buffer, B);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height, R, B,
		imgState.bucket_offsets,
		sampleState.bucket_to_tile,
		sampleState.T,
		sampleState.ar,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.max_contrib,
		imgState.pixel_colors,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug);

	// // Take care of the rest of preprocessing. Was the precomputed covariance
	// // given to us or a scales/rot pair? If precomputed, pass that. If not,
	// // use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		dc,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_ddc,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug);
}