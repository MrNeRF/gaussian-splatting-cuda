#include "core/knn_utils.hpp"
#include "core/torch_shapes.hpp" // For TORCH_CHECK macros if not implicitly included by torch/torch.h
#include <algorithm> // For std::sort, std::min, std::max
#include <vector>    // For std::vector
#include <cmath>     // For std::acos
// #include <map>    // Not needed with current approach

namespace gs {
namespace utils {

// Helper to ensure dot product is within acos valid range [-1.0, 1.0]
inline float clamp_dot_product(float dp) {
    return std::max(-1.0f, std::min(1.0f, dp));
}

// Projects a point onto the surface of a sphere.
torch::Tensor project_to_sphere(const torch::Tensor& point, const torch::Tensor& sphere_center, float sphere_radius) {
    // Ensure point and sphere_center are compatible
    TORCH_CHECK(point.sizes() == sphere_center.sizes(), "Point and sphere_center must have the same dimensions");
    TORCH_CHECK(point.dim() == 1, "Point and sphere_center must be 1D tensors (vectors)");

    if (torch::allclose(point, sphere_center)) {
        // Point is at the center of the sphere.
        // Project to an arbitrary point on the sphere, e.g., along Z-axis from center.
        // This case should be rare for camera positions relative to scene_center.
        torch::Tensor offset = torch::zeros_like(point);
        if (offset.size(0) > 2) { // Check if there is a Z-axis
             offset[2] = sphere_radius;
        } else { // For 2D or 1D, just use the first available axis or handle error
            offset[0] = sphere_radius;
        }
        return sphere_center + offset;
    }
    torch::Tensor vec_to_point = point - sphere_center;
    return sphere_center + torch::nn::functional::normalize(vec_to_point, torch::nn::functional::NormalizeFuncOptions().dim(0)) * sphere_radius;
}


std::vector<std::vector<int>> calculate_camera_knns(
    const torch::Tensor& camera_world_positions, // Shape [num_cameras, 3]
    const torch::Tensor& scene_center,           // Shape [3]
    float bounding_sphere_radius,
    int K_neighbors,
    const std::vector<int>& camera_uids          // UIDs for each row in camera_world_positions
) {
    TORCH_CHECK(camera_world_positions.dim() == 2 && camera_world_positions.size(1) == 3, "camera_world_positions must be [N, 3]");
    TORCH_CHECK(scene_center.dim() == 1 && scene_center.size(0) == 3, "scene_center must be [3]");
    TORCH_CHECK(camera_world_positions.device() == scene_center.device(), "camera_world_positions and scene_center must be on the same device");
    TORCH_CHECK(static_cast<int64_t>(camera_uids.size()) == camera_world_positions.size(0),
                "Number of camera_uids must match number of camera_world_positions");
    TORCH_CHECK(bounding_sphere_radius > 1e-6f, "Bounding sphere radius must be positive"); // Use a small epsilon for float comparison
    TORCH_CHECK(K_neighbors >= 0, "K_neighbors must be non-negative");

    int num_cameras = camera_world_positions.size(0);

    if (num_cameras == 0) {
        return {};
    }

    // Ensure K is not larger than the number of other available cameras
    if (num_cameras <= 1) { // No neighbors if only one camera or zero
        K_neighbors = 0;
    } else {
        K_neighbors = std::min(K_neighbors, num_cameras - 1);
    }

    if (K_neighbors == 0 && num_cameras > 1) { // If K was 0 but there are other cameras, effectively means no KNNs desired.
         // This is a valid case, user might want 0 neighbors.
    } else if (K_neighbors == 0 && num_cameras <=1) {
        // Also valid, no neighbors to find.
    }


    // Determine max_uid to size the output vector correctly for direct UID indexing
    int max_uid = 0;
    if (!camera_uids.empty()) {
        for (int uid : camera_uids) {
            if (uid < 0) {
                TORCH_CHECK(false, "Camera UIDs must be non-negative.");
            }
            if (uid > max_uid) max_uid = uid;
        }
    }

    std::vector<std::vector<int>> knns_by_uid(max_uid + 1);
    if (num_cameras == 0 || K_neighbors == 0) { // If no cameras or K=0, return empty (or correctly sized empty) KNN lists
        return knns_by_uid;
    }

    // Pre-project all camera positions to the sphere. Ensure tensors are on the same device.
    torch::Tensor projected_positions = torch::empty_like(camera_world_positions);
    for (int i = 0; i < num_cameras; ++i) {
        projected_positions[i] = project_to_sphere(camera_world_positions[i], scene_center, bounding_sphere_radius);
    }

    // Calculate normalized vectors from sphere center to projected points
    // Add unsqueeze(0) to scene_center to allow broadcasting: [N,3] - [1,3] -> [N,3]
    torch::Tensor vecs_from_center = projected_positions - scene_center.unsqueeze(0);
    torch::Tensor normalized_vecs_from_center = torch::nn::functional::normalize(vecs_from_center,
                                                                                 torch::nn::functional::NormalizeFuncOptions().dim(1).eps(1e-12));


    for (int i = 0; i < num_cameras; ++i) {
        const torch::Tensor& primary_norm_vec = normalized_vecs_from_center[i]; // This is a [3] tensor slice
        int primary_cam_uid = camera_uids[i];

        std::vector<std::pair<float, int>> distances_to_others;
        distances_to_others.reserve(num_cameras > 1 ? num_cameras - 1 : 0);

        for (int j = 0; j < num_cameras; ++j) {
            if (i == j) continue; // Skip self

            const torch::Tensor& secondary_norm_vec = normalized_vecs_from_center[j]; // This is a [3] tensor slice
            int secondary_cam_uid = camera_uids[j];

            // Ensure dot product is between tensors on the same device.
            // primary_norm_vec and secondary_norm_vec are slices from normalized_vecs_from_center, so they are.
            float dot_product = torch::dot(primary_norm_vec, secondary_norm_vec).item<float>();
            float angle = std::acos(clamp_dot_product(dot_product)); // Clamp dot_product to avoid acos domain errors
            float spherical_distance = bounding_sphere_radius * angle;

            distances_to_others.emplace_back(spherical_distance, secondary_cam_uid);
        }

        if (!distances_to_others.empty()) {
             std::sort(distances_to_others.begin(), distances_to_others.end(), [](const auto& a, const auto& b) {
                return a.first < b.first; // Sort by distance, ascending
            });

            knns_by_uid[primary_cam_uid].reserve(K_neighbors);
            for (int k = 0; k < K_neighbors && k < static_cast<int>(distances_to_others.size()); ++k) {
                knns_by_uid[primary_cam_uid].push_back(distances_to_others[k].second);
            }
        }
    }
    return knns_by_uid;
}

} // namespace utils
} // namespace gs
