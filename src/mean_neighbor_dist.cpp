/*
 * Simple CPU-based k-nearest neighbor distance computation
 * No external dependencies, reasonable performance for typical datasets
 */

#include "core/mean_neighbor_dist.hpp"
#include <algorithm>
#include <cmath>
#include <torch/torch.h>
#include <vector>

torch::Tensor compute_mean_neighbor_distances(const torch::Tensor& points) {
    // Move to CPU for processing
    auto cpu_points = points.to(torch::kCPU).contiguous();
    const int num_points = cpu_points.size(0);

    // Validate input
    TORCH_CHECK(cpu_points.dim() == 2 && cpu_points.size(1) == 3,
                "Input points must have shape [N, 3]");
    TORCH_CHECK(cpu_points.dtype() == torch::kFloat32,
                "Input points must be float32");

    // Handle edge cases
    if (num_points <= 1) {
        return torch::full({num_points}, 0.01f, points.options());
    }

    // Get raw data pointer for efficient access
    const float* data = cpu_points.data_ptr<float>();

    // Allocate result
    auto result = torch::zeros({num_points}, torch::kFloat32);
    float* result_data = result.data_ptr<float>();

// Process each point
#pragma omp parallel for if (num_points > 1000)
    for (int i = 0; i < num_points; i++) {
        // Current point coordinates
        float px = data[i * 3 + 0];
        float py = data[i * 3 + 1];
        float pz = data[i * 3 + 2];

        // Find 3 nearest neighbors using partial sorting
        std::vector<float> distances;
        distances.reserve(num_points - 1);

        // Compute distances to all other points
        for (int j = 0; j < num_points; j++) {
            if (i == j)
                continue; // Skip self

            float dx = px - data[j * 3 + 0];
            float dy = py - data[j * 3 + 1];
            float dz = pz - data[j * 3 + 2];
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

            distances.push_back(dist);
        }

        // Find mean of 3 smallest distances
        if (distances.size() >= 3) {
            std::partial_sort(distances.begin(), distances.begin() + 3, distances.end());
            result_data[i] = (distances[0] + distances[1] + distances[2]) / 3.0f;
        } else if (distances.size() > 0) {
            // Less than 3 neighbors, use what we have
            float sum = 0.0f;
            for (float d : distances) {
                sum += d;
            }
            result_data[i] = sum / distances.size();
        } else {
            result_data[i] = 0.01f; // Fallback
        }
    }

    // Return on original device
    return result.to(points.device());
}