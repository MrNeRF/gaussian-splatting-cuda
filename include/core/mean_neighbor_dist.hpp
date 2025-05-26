/*
 * Spatial operations using nanoflann for efficient k-nearest neighbor search
 * Used for initializing Gaussian scaling parameters in 3D Gaussian Splatting
 */

#pragma once
#include <torch/extension.h>

// Compute mean distance to 3 nearest neighbors for each point using nanoflann
// Input: points tensor of shape [N, 3] containing 3D coordinates
// Output: tensor of shape [N] with mean distances to 3 nearest neighbors
torch::Tensor compute_mean_neighbor_distances(const torch::Tensor& points);
