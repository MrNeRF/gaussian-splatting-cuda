#pragma once
#include <cstdio>
#include <string>
#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor& img1,
    torch::Tensor& img2,
    bool train);

torch::Tensor
fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor& img1,
    torch::Tensor& img2,
    torch::Tensor& dL_dmap,
    torch::Tensor& dm_dmu1,
    torch::Tensor& dm_dsigma1_sq,
    torch::Tensor& dm_dsigma12);
