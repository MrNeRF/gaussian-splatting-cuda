#include <ATen/Dispatch.h> // AT_DISPATCH_XXX
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h> // at::cuda::getCurrentCUDAStream
#include <cooperative_groups.h>

#include "Null.h"

static const uint8_t MASK_COUNTER_THRESHOLD = 16;

namespace gsplat {

namespace cg = cooperative_groups;

__global__ void process_mask_counter(
    uint8_t *__restrict__ mask_counter,
    const bool *__restrict__ valid,
    const uint32_t N
) {
    auto g_idx = cg::this_grid().thread_rank();
    if (g_idx >= N)
        return;
    // mask_counter:
    // 0: initial state
    // <=MASK_COUNTER_THRESHOLD: masked
    // >MASK_COUNTER_THRESHOLD: unmasked
    if (mask_counter[g_idx] >= MASK_COUNTER_THRESHOLD)
        mask_counter[g_idx] = 0;
    if (valid != nullptr && !valid[g_idx]) {
        mask_counter[g_idx] += 1;
    } else {
        mask_counter[g_idx] += 1 + MASK_COUNTER_THRESHOLD;
    }
}

template <typename scalar_t>
__global__ void adam_kernel_fused(
    const uint32_t N,
    const uint32_t D,
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ param_grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    const uint8_t* __restrict__ mask_counter,
    const int64_t step_count,
    const float lr,
    const float b1,
    const float b2,
    const float eps) {
    auto p_idx = cg::this_grid().thread_rank();
    const uint32_t g_idx = p_idx / D;

    if (g_idx >= N)
        return;

    uint8_t register_counter = mask_counter ? mask_counter[g_idx] : 1 + MASK_COUNTER_THRESHOLD;
    if (register_counter < MASK_COUNTER_THRESHOLD)
        return;
    if (register_counter > MASK_COUNTER_THRESHOLD)
        register_counter -= MASK_COUNTER_THRESHOLD;
    float register_param_grad = param_grad[p_idx];
    float register_exp_avg = exp_avg[p_idx];
    float register_exp_avg_sq = exp_avg_sq[p_idx];
    float bias_correction1 = 1 - powf(b1, step_count);
    float bias_correction2 = 1 - powf(b2, step_count);
    // if (isnan(register_param_grad)) {
    //     printf("Grad N: %d, D: %d\n", N, D);
    // }
    // last step * (k + k^2 + k^3 + ... + k^register_counter) ==
    // last step * k * (1 - k^register_counter) / (1 - k)
    float k = b1 / sqrtf(b2);
    float masked_steps = register_exp_avg / (sqrt(register_exp_avg_sq) / sqrt(bias_correction2) + eps) *
                         k * (1 - powf(k, register_counter - 1)) / (1 - k);
    register_exp_avg =
        powf(b1, register_counter) * register_exp_avg + (1.0f - b1) * register_param_grad;
    register_exp_avg_sq = powf(b2, register_counter) * register_exp_avg_sq +
                          (1.0f - b2) * register_param_grad * register_param_grad;
    float step = register_exp_avg / (sqrt(register_exp_avg_sq) / sqrt(bias_correction2) + eps);

    param[p_idx] += (step + masked_steps) * -lr / bias_correction1;
    exp_avg[p_idx] = register_exp_avg;
    exp_avg_sq[p_idx] = register_exp_avg_sq;
    // if (isnan(param[p_idx])) {
    //     printf("Param N: %d, D: %d\n", N, D);
    // }
}

template <typename scalar_t>
__global__ void adam_kernel_unrolled(
    const uint32_t N,
    const uint32_t D,
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ param_grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    const uint8_t* __restrict__ mask_counter,
    const int64_t step_count,
    const float lr,
    const float b1,
    const float b2,
    const float eps) {
    auto p_idx = cg::this_grid().thread_rank();
    const uint32_t g_idx = p_idx / D;

    if (g_idx >= N)
        return;

    uint8_t register_counter = mask_counter ? mask_counter[g_idx] : 1 + MASK_COUNTER_THRESHOLD;
    if (register_counter < MASK_COUNTER_THRESHOLD)
        return;
    if (register_counter > MASK_COUNTER_THRESHOLD)
        register_counter -= MASK_COUNTER_THRESHOLD;

    float register_param_grad = param_grad[p_idx];
    float register_exp_avg = exp_avg[p_idx];
    float register_exp_avg_sq = exp_avg_sq[p_idx];
    float register_param = param[p_idx];
    for (int i = register_counter - 1; i >= 0; i--) {
        float bias_correction1 = 1 - powf(b1, step_count - i);
        float bias_correction2 = 1 - powf(b2, step_count - i);
        register_exp_avg = b1 * register_exp_avg;
        register_exp_avg_sq = b2 * register_exp_avg_sq;
        if (i == 0) {
            register_exp_avg += (1.0f - b1) * register_param_grad;
            register_exp_avg_sq += (1.0f - b2) * register_param_grad * register_param_grad;
        }
        register_param -= lr / bias_correction1 * register_exp_avg
                          / (sqrt(register_exp_avg_sq) / sqrt(bias_correction2) + eps);
    }
    param[p_idx] = register_param;
    exp_avg[p_idx] = register_exp_avg;
    exp_avg_sq[p_idx] = register_exp_avg_sq;
}

void launch_adam_kernel(
    at::Tensor& param,                    // [N, ...]
    const at::Tensor& param_grad,         // [N, ...]
    at::Tensor& exp_avg,                  // [N, ...]
    at::Tensor& exp_avg_sq,               // [N, ...]
    at::Tensor& mask_counter,             // [N, ...]
    const at::optional<at::Tensor> valid, // [N]
    const int64_t step_count,
    const float lr,
    const float b1,
    const float b2,
    const float eps) {
    const uint32_t N = param.size(0);
    const uint32_t D = param.numel() / N;

    // parallel over [N, ...]
    int64_t n_elements = N * D;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    if (valid.has_value()) {
        dim3 mask_counter_grid((N + threads.x - 1) / threads.x);
        process_mask_counter<<<
            mask_counter_grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            mask_counter.data_ptr<uint8_t>(),
            valid.has_value() ? valid.value().data_ptr<bool>() : nullptr,
            N);
    }

    AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "adam_kernel", [&]() {
        adam_kernel_fused<scalar_t>
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                N,
                D,
                param.data_ptr<scalar_t>(),
                param_grad.data_ptr<scalar_t>(),
                exp_avg.data_ptr<scalar_t>(),
                exp_avg_sq.data_ptr<scalar_t>(),
                valid.has_value() ? mask_counter.data_ptr<uint8_t>() : nullptr,
                step_count,
                lr,
                b1,
                b2,
                eps);
    });
}

} // namespace gsplat
