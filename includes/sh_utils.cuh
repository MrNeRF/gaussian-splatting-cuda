// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include <torch/torch.h>

static const double C0 = 0.28209479177387814;
static const double C1 = 0.4886025119029199;
inline torch::Tensor RGB2SH(const torch::Tensor& rgb) {
    return (rgb - 0.5) / C0;
}

inline torch::Tensor SH2RGB(const torch::Tensor& sh) {
    return sh * C0 + 0.5;
}
