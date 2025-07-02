#include "core/setup_utils.hpp"
#include "core/knn_utils.hpp" // For calculate_camera_knns
#include "core/splat_data.hpp"
#include "core/dataset.hpp"
#include "core/parameters.hpp"
#include <torch/torch.h>
#include <vector>
#include <iostream> // For std::cout, std::cerr
#include <chrono>   // For time tracking

namespace gs {
namespace utils {

void setup_camera_knn_for_splat_data(
    SplatData& splat_data,
    const std::shared_ptr<CameraDataset>& dataset,
    const torch::Tensor& camera_world_positions,
    const torch::Tensor& scene_center,
    const gs::param::OptimizationParameters& opt_params
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "INFO: Setting up camera KNN data..." << std::endl;

    // 1. Get K_neighbors from optimization parameters
    const int K_neighbors = opt_params.newton_knn_k;
    if (K_neighbors <= 0) {
        std::cout << "K_neighbors (opt_params.newton_knn_k) is " << K_neighbors << ". Skipping camera KNN calculation." << std::endl;
        // Ensure KNN list is empty or appropriately sized if K=0
        splat_data.set_camera_knns({}); // Set to empty KNNs
        return;
    }

    // 2. Calculate bounding_sphere_radius from SplatData means and scene_center
    torch::Tensor gs_means = splat_data.get_means(); // Typically on CUDA
    torch::Tensor scene_center_device = scene_center.to(gs_means.device()); // Move scene_center to device of gs_means

    float bounding_sphere_radius;
    if (gs_means.size(0) == 0) {
        std::cerr << "Warning: No GS means found to calculate bounding sphere for KNN. Using scene_scale or fallback 1.0." << std::endl;
        bounding_sphere_radius = splat_data.get_scene_scale(); // Fallback to scene_scale
        if (bounding_sphere_radius <= 1e-6f) { // Further fallback if scene_scale is also invalid
            bounding_sphere_radius = 1.0f;
        }
    } else {
        torch::Tensor distances_to_center = torch::norm(gs_means - scene_center_device.unsqueeze(0), 2, 1);
        bounding_sphere_radius = torch::max(distances_to_center).item<float>();
    }

    if (bounding_sphere_radius <= 1e-6f) { // Final check and fallback for calculated radius
        std::cout << "Warning: Calculated bounding_sphere_radius for KNN is very small or zero (" << bounding_sphere_radius << "). Using fallback radius 1.0." << std::endl;
        bounding_sphere_radius = 1.0f;
    }
    std::cout << "  - Bounding sphere radius for KNN: " << bounding_sphere_radius << std::endl;

    // 3. Collect camera UIDs from the dataset
    std::vector<int> camera_uids;
    const auto& cameras_in_dataset = dataset->get_cameras();
    camera_uids.reserve(cameras_in_dataset.size());
    for (const auto& cam_ptr : cameras_in_dataset) {
        camera_uids.push_back(cam_ptr->uid());
    }

    if (camera_uids.empty()) {
        std::cerr << "Warning: No camera UIDs found in dataset. Skipping KNN calculation." << std::endl;
        splat_data.set_camera_knns({}); // Set to empty KNNs
        return;
    }
    if (!camera_world_positions.defined() || camera_world_positions.size(0) == 0) {
        std::cerr << "Warning: camera_world_positions tensor is not defined or empty. Skipping KNN calculation." << std::endl;
        splat_data.set_camera_knns({}); // Set to empty KNNs
        return;
    }
     if (static_cast<size_t>(camera_world_positions.size(0)) != camera_uids.size()) {
        std::cerr << "Error: Mismatch between number of camera world positions (" << camera_world_positions.size(0)
                  << ") and camera UIDs (" << camera_uids.size() << "). Skipping KNN calculation." << std::endl;
        splat_data.set_camera_knns({});
        return;
    }


    // 4. Call gs::utils::calculate_camera_knns
    // Note: calculate_camera_knns expects camera_world_positions and scene_center on potentially any device,
    // and it handles internal tensor operations.
    // The current implementation of calculate_camera_knns uses CPU for some std::sort, etc.
    // so it might be beneficial to ensure inputs are on CPU if they aren't used on GPU inside.
    // For now, pass as is; knn_utils has device checks.
    std::cout << "  - Calculating KNNs with K=" << K_neighbors << " for " << camera_uids.size() << " cameras." << std::endl;
    std::vector<std::vector<int>> knns = gs::utils::calculate_camera_knns(
        camera_world_positions, //.to(torch::kCPU) // Optional: move to CPU if calculate_camera_knns is mostly CPU bound
        scene_center,           //.to(torch::kCPU) // Optional: move to CPU
        bounding_sphere_radius,
        K_neighbors,
        camera_uids
    );

    // 5. Call splat_data.set_camera_knns()
    splat_data.set_camera_knns(std::move(knns));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "INFO: Camera KNN data setup complete. Took " << duration.count() << " ms." << std::endl;
}

} // namespace utils
} // namespace gs
