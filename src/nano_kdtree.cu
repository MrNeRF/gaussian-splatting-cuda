//
// Created by paja on 8/30/24.
//

#include "nano_kdtree.cuh"

torch::Tensor Nano_kdtree::compute_scales() const {
    // Ensure the KD-Tree is created if not already present.
    auto index = const_cast<Nano_kdtree*>(this)->ensureKdTree<KdTreeTensor>();

    torch::Tensor scales = torch::zeros({_xyz.size(0), 1}, torch::kFloat32);

    const int num_neighbors = 4; // Number of neighbors to consider.
    std::vector<size_t> neighbor_indices(num_neighbors);
    std::vector<float> neighbor_distances(num_neighbors);

    for (size_t i = 0; i < _xyz.size(0); ++i) {
        // Perform KNN search.
        index->knnSearch(reinterpret_cast<float*>(_xyz[i].data_ptr()), num_neighbors,
                         neighbor_indices.data(), neighbor_distances.data());

        // Calculate the average distance to the nearest neighbors.
        float sum_distances = 0.0f;
        for (size_t j = 1; j < num_neighbors; ++j) {
            sum_distances += std::sqrt(neighbor_distances[j]);
        }
        scales[i][0] = sum_distances / (num_neighbors - 1);
    }

    return scales;
}
