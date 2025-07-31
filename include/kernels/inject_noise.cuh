#pragma once
#include <ATen/core/Tensor.h>

void launch_get_opacity_sigmoid(
    const at::Tensor& opacity,
    at::Tensor& opacity_sigmoid,
    int num);

void launch_sgemv_3x3(
    const at::Tensor& A, // [N, 3, 3]
    const at::Tensor& B, // [N, 3]
    at::Tensor& C       // [N, 3]
);