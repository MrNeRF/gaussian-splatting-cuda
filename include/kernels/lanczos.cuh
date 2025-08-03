/*
Copyright (c) 2025 Youyu Chen
SPDX-License-Identifier: MIT
*/
#pragma once
#include <torch/extension.h>

#define BLOCK_X 16
#define BLOCK_Y 16
#define NUM_CHANNELS 3

torch::Tensor LanczosResampling(
	const torch::Tensor &input, 
	const int output_h, 
	const int output_w, 
	const int kernel_size
);