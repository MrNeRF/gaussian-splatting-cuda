#pragma once

#include <torch/torch.h>
#include <vector>
// #include <memory> // Not strictly needed for this function signature

// Forward declaration of Camera class is not needed if we pass camera_world_positions tensor

namespace gs { // Assuming a namespace like gs for Gaussian Splatting project
namespace utils {

/**
 * @brief Calculates K-Nearest Neighbors for each camera based on spherical distance.
 *
 * Projects camera world positions onto a bounding sphere and computes spherical distances
 * between these projections to find the K nearest neighbors for each camera.
 *
 * @param camera_world_positions Tensor of shape [num_cameras, 3] containing world coordinates of each camera.
 * @param scene_center Tensor of shape [3] representing the center of the bounding sphere.
 * @param bounding_sphere_radius Radius of the bounding sphere.
 * @param K_neighbors The number of nearest neighbors to find for each camera.
 * @param camera_uids A vector of UIDs corresponding to each row in camera_world_positions.
 *                    The UIDs are used to structure the output.
 * @return std::vector<std::vector<int>> A list where index `uid` holds a vector of UIDs
 *                                       of the K nearest neighbors for the camera with that `uid`.
 *                                       The outer vector is sized based on `max(camera_uids) + 1`.
 */
std::vector<std::vector<int>> calculate_camera_knns(
    const torch::Tensor& camera_world_positions, // Tensor of shape [num_cameras, 3]
    const torch::Tensor& scene_center,           // Tensor of shape [3]
    float bounding_sphere_radius,
    int K_neighbors,
    const std::vector<int>& camera_uids          // UIDs corresponding to rows in camera_world_positions
);

} // namespace utils
} // namespace gs
