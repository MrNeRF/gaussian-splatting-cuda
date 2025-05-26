/*
 * Efficient k-nearest neighbor distance computation using nanoflann
 */

#include "core/mean_neighbor_dist.hpp"
#include "external/nanoflann.hpp"
#include <algorithm>
#include <cmath>
#include <torch/torch.h>
#include <vector>

// Point cloud adaptor for nanoflann
struct PointCloudAdaptor {
    const float* points;
    size_t num_points;

    PointCloudAdaptor(const float* pts, size_t n) : points(pts),
                                                    num_points(n) {}

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return num_points; }

    // Returns the dim'th component of the idx'th point in the class
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return points[idx * 3 + dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

// Define the KD-tree type
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>,
    PointCloudAdaptor,
    3 /* dimensionality */
    >;

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

    // Create point cloud adaptor and build KD-tree
    PointCloudAdaptor cloud(data, num_points);
    KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();

    // Allocate result
    auto result = torch::zeros({num_points}, torch::kFloat32);
    float* result_data = result.data_ptr<float>();

// Process each point
#pragma omp parallel for if (num_points > 1000)
    for (int i = 0; i < num_points; i++) {
        // Query point
        const float query_pt[3] = {data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]};

        // We need 4 nearest neighbors (including self) to get 3 actual neighbors
        const size_t num_results = std::min(4, num_points);
        std::vector<size_t> ret_indices(num_results);
        std::vector<float> out_dists_sqr(num_results);

        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(&ret_indices[0], &out_dists_sqr[0]);
        index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParameters(10));

        // Calculate mean distance to 3 nearest neighbors (excluding self at distance 0)
        float sum_dist = 0.0f;
        int valid_neighbors = 0;

        for (size_t j = 0; j < num_results && valid_neighbors < 3; j++) {
            if (out_dists_sqr[j] > 1e-8f) { // Skip self (distance ~0)
                sum_dist += std::sqrt(out_dists_sqr[j]);
                valid_neighbors++;
            }
        }

        if (valid_neighbors > 0) {
            result_data[i] = sum_dist / valid_neighbors;
        } else {
            result_data[i] = 0.01f; // Fallback for edge cases
        }
    }

    // Return on original device
    return result.to(points.device());
}