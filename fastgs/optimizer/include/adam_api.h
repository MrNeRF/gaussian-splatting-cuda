/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <torch/torch.h>

namespace fast_gs::optimizer {

    void adam_step_wrapper(
        torch::Tensor& param,
        torch::Tensor& exp_avg,
        torch::Tensor& exp_avg_sq,
        const torch::Tensor& param_grad,
        const float lr,
        const float beta1,
        const float beta2,
        const float eps,
        const float bias_correction1,
        const float bias_correction2_sqrt);

}
