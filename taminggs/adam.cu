#include "auxiliary.h"
#include "adam.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// step on a grid of size (N, M)
// N is always number of gaussians
__global__
void adamUpdateCUDA(
    float* __restrict__ param,
    const float* __restrict__ param_grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const bool* tiles_touched,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M) {

	auto p_idx = cg::this_grid().thread_rank();
    const uint32_t g_idx = p_idx / M;
    if (g_idx >= N) return;
    if (tiles_touched[g_idx]) {
        float Register_param_grad = param_grad[p_idx];
        float Register_exp_avg = exp_avg[p_idx];
        float Register_exp_avg_sq = exp_avg_sq[p_idx];
        Register_exp_avg = b1 * Register_exp_avg + (1.0f - b1) * Register_param_grad;
        Register_exp_avg_sq = b2 * Register_exp_avg_sq + (1.0f - b2) * Register_param_grad * Register_param_grad;
        float step = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);

        param[p_idx] += step;
        exp_avg[p_idx] = Register_exp_avg;
        exp_avg_sq[p_idx] = Register_exp_avg_sq;
    }
}

void ADAM::adamUpdate(
    float* param,
    const float* param_grad,
    float* exp_avg,
    float* exp_avg_sq,
    const bool* tiles_touched,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M) {

    const uint32_t cnt = N * M;
    adamUpdateCUDA<<<(cnt + 255) / 256, 256>>> (
        param,
        param_grad,
        exp_avg,
        exp_avg_sq,
        tiles_touched,
        lr,
        b1,
        b2,
        eps,
        N,
        M
    );
}