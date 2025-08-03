#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

// for CUB_WRAPPER
#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>

#include "Common.h"
#include "Intersect.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

// Precise ellipsoid-to-tile intersection helper functions
__forceinline__ __device__ bool segment_intersect_ellipse(float a, float b, float c, float d, float l, float r)
{
    float delta = b * b - 4.0f * a * c;
    float t1 = (l - d) * (2.0f * a) + b;
    float t2 = (r - d) * (2.0f * a) + b;
    return delta >= 0.0f && (t1 <= 0.0f || t1 * t1 <= delta) && (t2 >= 0.0f || t2 * t2 <= delta);
}

__forceinline__ __device__ bool block_intersect_ellipse(int2 pix_min, int2 pix_max, float2 center, float3 conic, float power)
{
    float a, b, c, dx, dy;
    float w = 2.0f * power;

    // Test intersection with rectangle edges
    if (center.x * 2.0f < pix_min.x + pix_max.x) {
        dx = center.x - pix_min.x;
    } else {
        dx = center.x - pix_max.x;
    }
    a = conic.z;
    b = -2.0f * conic.y * dx;
    c = conic.x * dx * dx - w;

    if (segment_intersect_ellipse(a, b, c, center.y, pix_min.y, pix_max.y)) {
        return true;
    }

    if (center.y * 2.0f < pix_min.y + pix_max.y) {
        dy = center.y - pix_min.y;
    } else {
        dy = center.y - pix_max.y;
    }
    a = conic.x;
    b = -2.0f * conic.y * dy;
    c = conic.z * dy * dy - w;

    if (segment_intersect_ellipse(a, b, c, center.x, pix_min.x, pix_max.x)) {
        return true;
    }

    return false;
}

__forceinline__ __device__ bool block_contains_center(int2 pix_min, int2 pix_max, float2 center)
{
    return center.x >= pix_min.x && center.x <= pix_max.x && 
           center.y >= pix_min.y && center.y <= pix_max.y;
}

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation

template <typename scalar_t>
__global__ void intersect_tile_kernel(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const scalar_t *__restrict__ means2d,            // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N, 2] or [nnz, 2]
    const scalar_t *__restrict__ depths,             // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    vec2 mean2d = glm::make_vec2(means2d + 2 * idx);

    float tile_radius_x = radius_x / static_cast<float>(tile_size);
    float tile_radius_y = radius_y / static_cast<float>(tile_size);
    float tile_x = mean2d.x / static_cast<float>(tile_size);
    float tile_y = mean2d.y / static_cast<float>(tile_size);

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius_x)), tile_width);
    tile_min.y =
        min(max(0, (uint32_t)floor(tile_y - tile_radius_y)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius_x)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius_y)), tile_height);

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] = static_cast<int32_t>(
            (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
        );
        return;
    }

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
        // gid = idx % N;
    }
    const int64_t cid_enc = cid << (32 + tile_n_bits);

    // tolerance for negative depth
    int32_t depth_i32 = *(int32_t *)&(depths[idx]);  // Bit-level reinterpret
    int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);  // Zero-extend to 64-bit
    // int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * tile_width + j;
            // e.g. tile_n_bits = 22:
            // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [C * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}

// Precise ellipsoid-to-tile intersection testing
template <typename scalar_t>
__global__ void intersect_tile_kernel_precise(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const scalar_t *__restrict__ means2d,            // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N, 2] or [nnz, 2]
    const scalar_t *__restrict__ depths,             // [C, N] or [nnz]
    const scalar_t *__restrict__ conics,             // [C, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ opacities,          // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    vec2 mean2d = glm::make_vec2(means2d + 2 * idx);

    float tile_radius_x = radius_x / static_cast<float>(tile_size);
    float tile_radius_y = radius_y / static_cast<float>(tile_size);
    float tile_x = mean2d.x / static_cast<float>(tile_size);
    float tile_y = mean2d.y / static_cast<float>(tile_size);

    // tile_min is inclusive, tile_max is exclusive (bounding box for initial culling)
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius_x)), tile_width);
    tile_min.y = min(max(0, (uint32_t)floor(tile_y - tile_radius_y)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius_x)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius_y)), tile_height);

    // Get conic and opacity for precise intersection testing
    const float3 conic = {
        static_cast<float>(conics[idx * 3]),
        static_cast<float>(conics[idx * 3 + 1]),
        static_cast<float>(conics[idx * 3 + 2])
    };
    const float opacity = static_cast<float>(opacities[idx]);
    const float power = -0.5f * log(opacity + 1e-8f);  // Convert to power parameter

    if (first_pass) {
        // First pass: count tiles with precise intersection testing
        int32_t tile_count = 0;
        for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
            for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
                // Calculate tile pixel bounds
                const int2 pix_min = {static_cast<int>(j * tile_size), static_cast<int>(i * tile_size)};
                const int2 pix_max = {pix_min.x + static_cast<int>(tile_size) - 1, pix_min.y + static_cast<int>(tile_size) - 1};
                const float2 center = {mean2d.x, mean2d.y};
                
                // Precise intersection test
                bool intersects = block_contains_center(pix_min, pix_max, center) ||
                                 block_intersect_ellipse(pix_min, pix_max, center, conic, power);
                
                if (intersects) {
                    tile_count++;
                }
            }
        }
        tiles_per_gauss[idx] = tile_count;
        return;
    }

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
        // gid = idx % N;
    }
    const int64_t cid_enc = cid << (32 + tile_n_bits);

    // tolerance for negative depth
    int32_t depth_i32 = *(int32_t *)&(depths[idx]);  // Bit-level reinterpret
    int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);  // Zero-extend to 64-bit
    
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    
    // Second pass: generate intersection data with precise testing
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            // Calculate tile pixel bounds
            const int2 pix_min = {static_cast<int>(j * tile_size), static_cast<int>(i * tile_size)};
            const int2 pix_max = {pix_min.x + static_cast<int>(tile_size) - 1, pix_min.y + static_cast<int>(tile_size) - 1};
            const float2 center = {mean2d.x, mean2d.y};
            
            // Precise intersection test
            bool intersects = block_contains_center(pix_min, pix_max, center) ||
                             block_intersect_ellipse(pix_min, pix_max, center, conic, power);
            
            if (intersects) {
                int64_t tile_id = i * tile_width + j;
                // e.g. tile_n_bits = 22:
                // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
                isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
                // the flatten index in [C * N] or [nnz]
                flatten_ids[cur_idx] = static_cast<int32_t>(idx);
                ++cur_idx;
            }
        }
    }
}

void launch_intersect_tile_kernel(
    // inputs
    const at::Tensor means2d,                    // [C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [C, N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [C, N] or [nnz]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [C, N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [C, N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids      // [n_isects]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N, nnz;
    int64_t n_elements;
    if (packed) {
        nnz = means2d.size(0); // total number of gaussians
        n_elements = nnz;
    } else {
        N = means2d.size(1); // number of gaussians per camera
        n_elements = C * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "intersect_tile_kernel",
        [&]() {
            intersect_tile_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids.has_value()
                        ? camera_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    gaussian_ids.has_value()
                        ? gaussian_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    means2d.data_ptr<scalar_t>(),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    cum_tiles_per_gauss.has_value()
                        ? cum_tiles_per_gauss.value().data_ptr<int64_t>()
                        : nullptr,
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    tiles_per_gauss.has_value()
                        ? tiles_per_gauss.value().data_ptr<int32_t>()
                        : nullptr,
                    isect_ids.has_value()
                        ? isect_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    flatten_ids.has_value()
                        ? flatten_ids.value().data_ptr<int32_t>()
                        : nullptr
                );
        }
    );
}

void launch_intersect_tile_kernel_precise(
    // inputs
    const at::Tensor means2d,                    // [C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [C, N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [C, N] or [nnz]
    const at::Tensor conics,                     // [C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                  // [C, N] or [nnz]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [C, N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [C, N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids      // [n_isects]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N, nnz;
    int64_t n_elements;
    if (packed) {
        nnz = means2d.size(0); // total number of gaussians
        n_elements = nnz;
    } else {
        N = means2d.size(1); // number of gaussians per camera
        n_elements = C * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "intersect_tile_kernel_precise",
        [&]() {
            intersect_tile_kernel_precise<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids.has_value()
                        ? camera_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    gaussian_ids.has_value()
                        ? gaussian_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    means2d.data_ptr<scalar_t>(),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    conics.data_ptr<scalar_t>(),     // ADD: conic sections
                    opacities.data_ptr<scalar_t>(),  // ADD: opacities
                    cum_tiles_per_gauss.has_value()
                        ? cum_tiles_per_gauss.value().data_ptr<int64_t>()
                        : nullptr,
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    tiles_per_gauss.has_value()
                        ? tiles_per_gauss.value().data_ptr<int32_t>()
                        : nullptr,
                    isect_ids.has_value()
                        ? isect_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    flatten_ids.has_value()
                        ? flatten_ids.value().data_ptr<int32_t>()
                        : nullptr
                );
        }
    );
}

__global__ void intersect_offset_kernel(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t C,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [C, n_tiles]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t cid_curr = isect_id_curr >> tile_n_bits;
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = cid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < C * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (cid,
        // tile_id) pair changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t cid_prev = isect_id_prev >> tile_n_bits;
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev = cid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

void launch_intersect_offset_kernel(
    // inputs
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [C, tile_height, tile_width]
) {
    int64_t n_elements = isect_ids.size(0); // total number of intersections
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        offsets.fill_(0);
        return;
    }

    uint32_t n_tiles = tile_width * tile_height;
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    intersect_offset_kernel<<<
        grid,
        threads,
        shmem_size,
        at::cuda::getCurrentCUDAStream()>>>(
        n_elements,
        isect_ids.data_ptr<int64_t>(),
        C,
        n_tiles,
        tile_n_bits,
        offsets.data_ptr<int32_t>()
    );
}

// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
// DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
void radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t tile_n_bits,
    const uint32_t cam_n_bits,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
) {
    if (n_isects <= 0) {
        return;
    }

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int64_t> d_keys(
        isect_ids.data_ptr<int64_t>(), isect_ids_sorted.data_ptr<int64_t>()
    );
    cub::DoubleBuffer<int32_t> d_values(
        flatten_ids.data_ptr<int32_t>(), flatten_ids_sorted.data_ptr<int32_t>()
    );
    CUB_WRAPPER(
        cub::DeviceRadixSort::SortPairs,
        d_keys,
        d_values,
        n_isects,
        0,
        32 + tile_n_bits + cam_n_bits,
        at::cuda::getCurrentCUDAStream()
    );
    switch (d_keys.selector) {
    case 0: // sorted items are stored in isect_ids
        isect_ids_sorted.set_(isect_ids);
        break;
    case 1: // sorted items are stored in isect_ids_sorted
        break;
    }
    switch (d_values.selector) {
    case 0: // sorted items are stored in flatten_ids
        flatten_ids_sorted.set_(flatten_ids);
        break;
    case 1: // sorted items are stored in flatten_ids_sorted
        break;
    }

    // Double buffer is better than naive radix sort, in terms of mem usage.
    // CUB_WRAPPER(
    //     cub::DeviceRadixSort::SortPairs,
    //     isect_ids,
    //     isect_ids_sorted,
    //     flatten_ids,
    //     flatten_ids_sorted,
    //     n_isects,
    //     0,
    //     32 + tile_n_bits + cam_n_bits,
    //     stream
    // );
}

} // namespace gsplat
