#pragma once

#include "core/splat_data.hpp" // For SplatData&
#include "core/dataset.hpp"     // For CameraDataset
#include "core/parameters.hpp"  // For OptimizationParams
#include <torch/torch.h>
#include <vector>
#include <memory> // For std::shared_ptr

namespace gs {
namespace utils {

/**
 * @brief Sets up K-Nearest Neighbor (KNN) information for cameras within SplatData.
 *
 * This function calculates the KNNs for each camera based on spherical distances
 * of their projections onto a scene bounding sphere. The results are then stored
 * in the provided SplatData object.
 *
 * @param splat_data Reference to the SplatData object where KNNs will be stored.
 * @param dataset Shared pointer to the CameraDataset, used to retrieve camera UIDs.
 * @param camera_world_positions Tensor of shape [num_cameras, 3] containing world coordinates.
 * @param scene_center Tensor of shape [3] for the bounding sphere center.
 * @param opt_params OptimizationParameters, used to retrieve K_neighbors.
 */
void setup_camera_knn_for_splat_data(
    SplatData& splat_data, // Non-const reference
    const std::shared_ptr<CameraDataset>& dataset,
    const torch::Tensor& camera_world_positions,
    const torch::Tensor& scene_center,
    const gs::param::OptimizationParameters& opt_params
);

} // namespace utils
} // namespace gs
