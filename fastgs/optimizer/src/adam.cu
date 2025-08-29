/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam.h"
#include "adam_kernels.cuh"
#include "optimizer_config.h"
#include "utils.h"

void fast_gs::optimizer::adam_step(
    float* param,
    float* exp_avg,
    float* exp_avg_sq,
    const float* param_grad,
    const int n_elements,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction1_rcp,
    const float bias_correction2_sqrt_rcp) {
    kernels::adam::adam_step_cu<<<div_round_up(n_elements, config::block_size_adam_step), config::block_size_adam_step>>>(
        param,
        exp_avg,
        exp_avg_sq,
        param_grad,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1_rcp,
        bias_correction2_sqrt_rcp);
    CHECK_CUDA(config::debug, "adam step")
}
