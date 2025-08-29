/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace fast_gs::optimizer::kernels::adam {

    // based on https://github.com/pytorch/pytorch/blob/9d32aa9789fc0ef0cad01a788157ecc2121db810/torch/csrc/api/src/optim/adam.cpp#L72-L142
    __global__ void adam_step_cu(
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
        auto idx = cg::this_grid().thread_rank();
        if (idx >= n_elements)
            return;
        const float grad = param_grad[idx];
        const float moment1 = beta1 * exp_avg[idx] + (1.0f - beta1) * grad;
        const float moment2 = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * grad * grad;
        const float denom = sqrtf(moment2) * bias_correction2_sqrt_rcp + eps;
        const float step_size = lr * bias_correction1_rcp;
        param[idx] -= step_size * moment1 / denom;
        exp_avg[idx] = moment1;
        exp_avg_sq[idx] = moment2;
    }

} // namespace fast_gs::optimizer::kernels::adam
