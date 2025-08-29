/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam.h"
#include "adam_api.h"

void fast_gs::optimizer::adam_step_wrapper(
    torch::Tensor& param,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const torch::Tensor& param_grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bias_correction1_rcp,
    const float bias_correction2_sqrt_rcp) {
    const int n_elements = param.numel();

    adam_step(
        param.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        param_grad.data_ptr<float>(),
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1_rcp,
        bias_correction2_sqrt_rcp);
}
