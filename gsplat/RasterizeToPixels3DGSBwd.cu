#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

static const int BACKGROUNDS_CONSTANT_SIZE = 256;
static __constant__ float backgrounds_constant[BACKGROUNDS_CONSTANT_SIZE];

static const int I_SIZE = 2;
static const int J_SIZE = 2;

__device__ static void wait_async_group();
__device__ static void copy_async_4(void* dst, const void* src);

template <uint32_t CDIM, typename scalar_t, uint32_t CDIM_PADDED = (6 + CDIM + 3) & ~3>
__launch_bounds__(256 / I_SIZE / J_SIZE, 8)
    __global__ void rasterize_to_pixels_3dgs_bwd_kernel(
        const uint32_t C,
        const uint32_t N,
        const uint32_t n_isects,
        const bool packed,
        // fwd inputs
        const float* __restrict__ gaussians, // [C, N, 6+CDIM] or [nnz, 6+CDIM]
        bool has_backgrounds,
        const bool* __restrict__ masks, // [C, tile_height, tile_width]
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const int32_t* __restrict__ tile_offsets, // [C, tile_height, tile_width]
        const int32_t* __restrict__ flatten_ids,  // [n_isects]
        // fwd outputs
        const scalar_t* __restrict__ render_alphas, // [C, image_height, image_width, 1]
        const int32_t* __restrict__ last_ids,       // [C, image_height, image_width]
        // grad outputs
        const scalar_t* __restrict__ v_render_colors, // [C, image_height,
                                                      // image_width, CDIM]
        const scalar_t* __restrict__ v_render_alphas, // [C, image_height, image_width, 1]
        // grad inputs
        float2* __restrict__ v_means2d_abs, // [C, N, 2] or [nnz, 2]
        float2* __restrict__ v_means2d,     // [C, N, 2] or [nnz, 2]
        float3* __restrict__ v_conics,      // [C, N, 3] or [nnz, 3]
        scalar_t* __restrict__ v_colors,    // [C, N, CDIM] or [nnz, CDIM]
        scalar_t* __restrict__ v_opacities  // [C, N] or [nnz]
    ) {
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i_base = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j_base = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    v_render_colors += camera_id * image_height * image_width * CDIM;
    v_render_alphas += camera_id * image_height * image_width;
    float *backgrounds = has_backgrounds ? backgrounds_constant + camera_id * CDIM : nullptr;
    float backgrounds_cache[CDIM];
    if (backgrounds != nullptr) {
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            backgrounds_cache[k] = backgrounds[k];
        }
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    float px[I_SIZE][J_SIZE];
    float py[I_SIZE][J_SIZE];
    int32_t pix_id[I_SIZE][J_SIZE];
    float T_final[I_SIZE][J_SIZE];
    float T[I_SIZE][J_SIZE];
    bool inside[I_SIZE][J_SIZE];
    float v_render_c[I_SIZE][J_SIZE][CDIM];
    float v_render_a[I_SIZE][J_SIZE];
    int32_t bin_final[I_SIZE][J_SIZE];
    int32_t thread_bin_final = 0;
    // the contribution from gaussians behind the current one
    float buffer[I_SIZE][J_SIZE][CDIM] = {0.f};
    for (int i = 0; i < I_SIZE; ++i) {
        for (int j = 0; j < J_SIZE; ++j) {
            int j_ = j_base + j * (tile_size / J_SIZE);
            int i_ = i_base + i * (tile_size / I_SIZE);
            px[i][j] = j_ + 0.5f;
            py[i][j] = i_ + 0.5f;
            // clamp this value to the last pixel
            pix_id[i][j] = min(i_ * image_width + j_, image_width * image_height - 1);
            // keep not rasterizing threads around for reading data
            inside[i][j] = (i_ < image_height && j_ < image_width);
            // this is the T AFTER the last gaussian in this pixel
            T_final[i][j] = 1.0f - render_alphas[pix_id[i][j]];
            T[i][j] = T_final[i][j];

            for (uint32_t k = 0; k < CDIM; ++k) {
                // df/d_out for this pixel
                v_render_c[i][j][k] = v_render_colors[pix_id[i][j] * CDIM + k];
            }
            v_render_a[i][j] = v_render_alphas[pix_id[i][j]];
            // index of last gaussian to contribute to this pixel
            bin_final[i][j] = inside[i][j] ? last_ids[pix_id[i][j]] : 0;
            thread_bin_final = max(thread_bin_final, bin_final[i][j]);
        }
    }
    auto warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, thread_bin_final, cg::greater<int>());
    if (warp_bin_final <= 0) // What if last_id == 0?
        return;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = warp_bin_final + 1;
    const uint32_t num_batches =
        (range_end - range_start + warp.size() - 1) / warp.size();

    extern __shared__ int s[];
    int32_t (*id_batch)[32] = (int32_t (*)[32])s;
    float (*gaussians_shared)[2][CDIM_PADDED] = (float (*)[2][CDIM_PADDED]) & id_batch[block.size() / 32];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    int w = warp.meta_group_rank();
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        warp.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end = range_end - 1 - warp.size() * b;
        const int32_t batch_size = min(warp.size(), batch_end + 1 - range_start);
        // Collectively load gaussians to coalesce global memory access
        const uint32_t tr = warp.thread_rank();
        if (batch_end >= range_start + tr) {
            id_batch[w][tr] = flatten_ids[batch_end - tr];
        } else {
            id_batch[w][tr] = -1;
        }
        warp.sync();
        assert(batch_size > 0);
        if (tr < 6 + CDIM) {
            gaussians_shared[w][0][tr] = gaussians[id_batch[w][0] * (6 + CDIM) + tr];
        }
        // wait for other threads to collect the gaussians in batch
        warp.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = 0; t < batch_size;
             ++t) {
            // Fetch next gaussian
            if (t + 1 < batch_size) {
                for (int32_t i = tr; i < 6 + CDIM; i += warp.size()) {
                    copy_async_4(&gaussians_shared[w][(t + 1) & 1][tr],
                                 &gaussians[id_batch[w][t + 1] * (6 + CDIM) + tr]);
                }
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            bool thread_valid = false;
            for (int i = 0; i < I_SIZE; i++) {
                for (int j = 0; j < J_SIZE; j++) {
                    bool valid = inside[i][j];
                    if (batch_end - t > bin_final[i][j]) {
                        valid = 0;
                    }
                    float alpha;
                    float opac;
                    vec2 delta;
                    float3 conic;
                    float vis;

                    const float4 part1 = *reinterpret_cast<const float4*>(&gaussians_shared[w][t & 1][0]);
                    const float2 part2 = *reinterpret_cast<const float2*>(&gaussians_shared[w][t & 1][4]);
                    conic = {part1.x, part1.y, part1.z};
                    float mean_x = part1.w;
                    float mean_y = part2.x;
                    opac = part2.y;
                    delta = {mean_x - px[i][j], mean_y - py[i][j]};
                    float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                    vis = __expf(-sigma);
                    alpha = min(0.999f, opac * vis);
                    if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                        valid = false;
                    }
                    thread_valid |= valid;

                    if (valid) {
                        // compute the current T for this gaussian
                        float ra = 1.0f / (1.0f - alpha);
                        T[i][j] *= ra;
                        // update v_rgb for this gaussian
                        const float fac = alpha * T[i][j];
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            v_rgb_local[k] += fac * v_render_c[i][j][k];
                        }
                        // contribution from this pixel
                        float v_alpha = 0.f;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            v_alpha += (gaussians_shared[w][t & 1][6 + k] * T[i][j] - buffer[i][j][k] * ra) *
                                       v_render_c[i][j][k];
                        }

                        v_alpha += T_final[i][j] * ra * v_render_a[i][j];
                        // contribution from background pixel
                        if (backgrounds != nullptr) {
                            float accum = 0.f;
#pragma unroll
                            for (uint32_t k = 0; k < CDIM; ++k) {
                                accum += backgrounds_cache[k] * v_render_c[i][j][k];
                            }
                            v_alpha += -T_final[i][j] * ra * accum;
                        }

                        if (opac * vis <= 0.999f) {
                            const float v_sigma = -opac * vis * v_alpha;
                            v_conic_local.x +=
                                0.5f * v_sigma * delta.x * delta.x;
                            v_conic_local.y += v_sigma * delta.x * delta.y;
                            v_conic_local.z += 0.5f * v_sigma * delta.y * delta.y;
                            v_xy_local.x +=
                                v_sigma * (conic.x * delta.x + conic.y * delta.y);
                            v_xy_local.y += v_sigma * (conic.y * delta.x + conic.z * delta.y);
                            if (v_means2d_abs != nullptr) {
                                v_xy_abs_local.x += abs(v_xy_local.x);
                                v_xy_abs_local.y += abs(v_xy_local.y);
                            }
                            v_opacity_local += vis * v_alpha;
                        }

#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            buffer[i][j][k] += gaussians_shared[w][t & 1][6 + k] * fac;
                        }
                    }
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if (warp.any(thread_valid)) {
                warpSum<CDIM>(v_rgb_local, warp);
                warpSum(v_conic_local, warp);
                warpSum(v_xy_local, warp);
                if (v_means2d_abs != nullptr) {
                    warpSum(v_xy_abs_local, warp);
                }
                warpSum(v_opacity_local, warp);
                if (warp.thread_rank() == 0) {
                    int32_t g = id_batch[w][t]; // flatten index in [C * N] or [nnz]
                    float* v_rgb_ptr = (float*)(v_colors) + CDIM * g;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                    }

                    float* v_conic_ptr = (float*)(v_conics) + 3 * g;
                    gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                    gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                    gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                    float* v_xy_ptr = (float*)(v_means2d) + 2 * g;
                    gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                    gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                    if (v_means2d_abs != nullptr) {
                        float* v_xy_abs_ptr = (float*)(v_means2d_abs) + 2 * g;
                        gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                        gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                    }

                    gpuAtomicAdd(v_opacities + g, v_opacity_local);
                }
            }
            wait_async_group();
            warp.sync();
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor gaussians,                 // [C, N, 9] or [nnz, 9]
    const at::optional<at::Tensor> backgrounds, // [C, 3]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [C, image_height, image_width, 1]
    const at::Tensor last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [C, image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [C, image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [C, N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [C, N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [C, N] or [nnz]
) {
    bool packed = false;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : gaussians.size(1); // number of gaussians
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size / J_SIZE, tile_size / I_SIZE, 1};
    dim3 grid = {C, tile_height, tile_width};

    int elem_size = sizeof(float) * ((6 * CDIM + 3) & ~3);
    int block_size = tile_size / I_SIZE * tile_size / J_SIZE;
    // Padded to 16 byte boundary
    int64_t shmem_size = block_size * sizeof(int32_t) + block_size / C10_WARP_SIZE * elem_size;

    if (n_isects == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    if (backgrounds.has_value()) {
        TORCH_CHECK(backgrounds->numel() < BACKGROUNDS_CONSTANT_SIZE);
        if (cudaMemcpyToSymbolAsync(backgrounds_constant,
                                    backgrounds->data_ptr<float>(),
                                    C * CDIM * sizeof(float), 0,
                                    cudaMemcpyDeviceToDevice,
                                    at::cuda::getCurrentCUDAStream()) != cudaSuccess) {
            AT_ERROR("Failed to copy backgrounds to constant memory");
        }
    }

    rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<float*>(gaussians.data_ptr<float>()),
            backgrounds.has_value(),
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_means2d_abs.has_value()
                ? reinterpret_cast<float2*>(
                      v_means2d_abs.value().data_ptr<float>())
                : nullptr,
            reinterpret_cast<float2*>(v_means2d.data_ptr<float>()),
            reinterpret_cast<float3*>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>());
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_3dgs_bwd_kernel<CDIM>(            \
        const at::Tensor gaussians,                                            \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const at::Tensor render_alphas,                                        \
        const at::Tensor last_ids,                                             \
        const at::Tensor v_render_colors,                                      \
        const at::Tensor v_render_alphas,                                      \
        at::optional<at::Tensor> v_means2d_abs,                                \
        at::Tensor v_means2d,                                                  \
        at::Tensor v_conics,                                                   \
        at::Tensor v_colors,                                                   \
        at::Tensor v_opacities                                                 \
    );

__INS__(1)
__INS__(2)
__INS__(3)
__INS__(4)
__INS__(5)
__INS__(8)
__INS__(9)
__INS__(16)
__INS__(17)
__INS__(32)
__INS__(33)
__INS__(64)
__INS__(65)
__INS__(128)
__INS__(129)
__INS__(256)
__INS__(257)
__INS__(512)
__INS__(513)
#undef __INS__

// Put last because syntax highlighting is not happy about them
__device__ static void wait_async_group() {
    asm("cp.async.commit_group;\n" ::);
    asm("cp.async.wait_group 0;\n" ::);
}

__device__ static void copy_async_4(void *dst, const void *src) {
    int dst_shared = __cvta_generic_to_shared(dst);
    asm("cp.async.ca.shared.global [%0], [%1], 4;\n" :
        : "r"(dst_shared), "l"(src));
    // *(float*)dst = *(const float*)src;
}

} // namespace gsplat
