/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "kernels/kmeans.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

namespace gs {
    namespace cuda {

        namespace {

            // Kernel to compute distances and assign labels
            template <int BLOCK_SIZE = 256>
            __global__ void assign_clusters_kernel(
                const float* __restrict__ data,
                const float* __restrict__ centroids,
                int* __restrict__ labels,
                float* __restrict__ distances,
                const int n_points,
                const int n_clusters,
                const int n_dims) {
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid >= n_points)
                    return;

                float min_dist = INFINITY;
                int min_idx = 0;

                // For each centroid
                for (int c = 0; c < n_clusters; ++c) {
                    float dist = 0.0f;

                    // Compute squared Euclidean distance
                    for (int d = 0; d < n_dims; ++d) {
                        float diff = data[tid * n_dims + d] - centroids[c * n_dims + d];
                        dist += diff * diff;
                    }

                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = c;
                    }
                }

                labels[tid] = min_idx;
                if (distances != nullptr) {
                    distances[tid] = min_dist;
                }
            }

            // Optimized kernel for 1D clustering
            __global__ void assign_clusters_1d_kernel(
                const float* __restrict__ data,
                const float* __restrict__ centroids,
                int* __restrict__ labels,
                const int n_points,
                const int n_clusters) {
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid >= n_points)
                    return;

                const float point = data[tid];
                float min_dist = INFINITY;
                int best = 0;

                // Linear search (sorted centroids)
                for (int c = 0; c < n_clusters; ++c) {
                    float dist = fabsf(point - centroids[c]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best = c;
                    }
                }

                labels[tid] = best;
            }

            // Kernel to update centroids
            __global__ void update_centroids_kernel(
                const float* __restrict__ data,
                const int* __restrict__ labels,
                float* __restrict__ new_centroids,
                int* __restrict__ counts,
                const int n_points,
                const int n_clusters,
                const int n_dims) {
                const int cluster_id = blockIdx.x;
                const int dim = threadIdx.x;

                if (cluster_id >= n_clusters || dim >= n_dims)
                    return;

                float sum = 0.0f;
                int count = 0;

                // Sum all points belonging to this cluster
                for (int i = 0; i < n_points; ++i) {
                    if (labels[i] == cluster_id) {
                        sum += data[i * n_dims + dim];
                        if (dim == 0)
                            count++;
                    }
                }

                // Store result
                if (dim == 0) {
                    counts[cluster_id] = count;
                }

                __syncthreads();

                if (counts[cluster_id] > 0) {
                    new_centroids[cluster_id * n_dims + dim] = sum / counts[cluster_id];
                }
            }

            // Initialize centroids using k-means++ algorithm
            torch::Tensor initialize_centroids_plusplus(
                const torch::Tensor& data,
                int k) {
                const int n = data.size(0);
                const int d = data.size(1);

                auto centroids = torch::zeros({k, d}, data.options());
                auto distances = torch::full({n}, INFINITY, data.options());

                // Choose first centroid randomly
                int first_idx = torch::randint(n, {1}, torch::kInt32).item<int>();
                centroids[0] = data[first_idx];

                // Choose remaining centroids
                for (int c = 1; c < k; ++c) {
                    // Compute distances to nearest centroid
                    auto centroid_view = centroids.slice(0, 0, c);
                    auto dists = torch::cdist(data, centroid_view);
                    distances = std::get<0>(torch::min(dists, 1));

                    // Choose next centroid with probability proportional to squared distance
                    auto probs = distances.pow(2);
                    probs = probs / probs.sum();

                    // Sample from distribution
                    auto cumsum = probs.cumsum(0);
                    float rand_val = torch::rand({1}).item<float>();
                    int next_idx = (cumsum >= rand_val).nonzero()[0].item<int>();

                    centroids[c] = data[next_idx];
                }

                return centroids;
            }

        } // anonymous namespace

        std::tuple<torch::Tensor, torch::Tensor> kmeans(
            const torch::Tensor& data,
            int k,
            int iterations,
            float tolerance) {
            TORCH_CHECK(data.dim() == 2, "Data must be 2D tensor [N, D]");
            TORCH_CHECK(data.is_cuda(), "Data must be on CUDA");
            TORCH_CHECK(data.dtype() == torch::kFloat32, "Data must be float32");

            const int n = data.size(0);
            const int d = data.size(1);

            if (n <= k) {
                // If fewer points than clusters, return points as centroids
                auto centroids = data.clone();
                auto labels = torch::arange(n, torch::kInt32).to(data.device());
                return {centroids, labels};
            }

            // Initialize centroids using k-means++
            auto centroids = initialize_centroids_plusplus(data, k);
            auto labels = torch::zeros({n}, torch::kInt32).to(data.device());
            auto old_centroids = torch::zeros_like(centroids);

            // Allocate workspace
            auto counts = torch::zeros({k}, torch::kInt32).to(data.device());

            const int block_size = 256;
            const int grid_size_points = (n + block_size - 1) / block_size;

            for (int iter = 0; iter < iterations; ++iter) {
                old_centroids.copy_(centroids);

                // Assign clusters
                assign_clusters_kernel<block_size><<<grid_size_points, block_size>>>(
                    data.data_ptr<float>(),
                    centroids.data_ptr<float>(),
                    labels.data_ptr<int>(),
                    nullptr,
                    n, k, d);

                // Update centroids
                counts.zero_();
                dim3 block(d, 1);
                dim3 grid(k, 1);
                update_centroids_kernel<<<grid, block>>>(
                    data.data_ptr<float>(),
                    labels.data_ptr<int>(),
                    centroids.data_ptr<float>(),
                    counts.data_ptr<int>(),
                    n, k, d);

                cudaDeviceSynchronize();

                // Check convergence
                float max_movement = (centroids - old_centroids).abs().max().item<float>();
                if (max_movement < tolerance) {
                    break;
                }
            }

            return {centroids, labels};
        }

        std::tuple<torch::Tensor, torch::Tensor> kmeans_1d(
            const torch::Tensor& data,
            int k,
            int iterations) {
            // Reshape to [N, 1] if needed
            auto data_2d = data.dim() == 1 ? data.unsqueeze(1) : data;

            TORCH_CHECK(data_2d.size(1) == 1, "kmeans_1d expects 1D data");

            const int n = data_2d.size(0);

            if (n <= k) {
                auto sorted = std::get<0>(data_2d.sort(0));
                auto labels = torch::arange(n, torch::kInt32).to(data.device());
                return {sorted, labels};
            }

            // For 1D, initialize centroids evenly across range
            auto [min_val, _] = data_2d.min(0);
            auto [max_val, __] = data_2d.max(0);

            auto centroids = torch::linspace(
                                 min_val.item<float>(),
                                 max_val.item<float>(),
                                 k,
                                 data_2d.options())
                                 .unsqueeze(1);

            auto labels = torch::zeros({n}, torch::kInt32).to(data.device());

            const int block_size = 256;
            const int grid_size = (n + block_size - 1) / block_size;

            for (int iter = 0; iter < iterations; ++iter) {
                // Sort centroids for efficient 1D assignment
                auto [sorted_centroids, sort_idx] = centroids.squeeze(1).sort(0);

                // Assign clusters using optimized 1D kernel
                assign_clusters_1d_kernel<<<grid_size, block_size>>>(
                    data_2d.data_ptr<float>(),
                    sorted_centroids.data_ptr<float>(),
                    labels.data_ptr<int>(),
                    n, k);

                cudaDeviceSynchronize();

                // Update centroids
                for (int c = 0; c < k; ++c) {
                    auto mask = labels == c;
                    if (mask.any().item<bool>()) {
                        auto cluster_points = data_2d.masked_select(mask.unsqueeze(1)).reshape({-1, 1});
                        centroids[c] = cluster_points.mean(0);
                    }
                }
            }

            // Final sort of centroids and remap labels
            auto [final_sorted, final_idx] = centroids.squeeze(1).sort(0);
            centroids = final_sorted.unsqueeze(1);

            // Create inverse mapping for labels
            auto inv_map = torch::zeros({k}, torch::kInt32).to(data.device());
            for (int i = 0; i < k; ++i) {
                inv_map[final_idx[i].item<int>()] = i;
            }

            // Remap labels
            auto remapped_labels = torch::zeros_like(labels);
            thrust::gather(
                thrust::device,
                labels.data_ptr<int>(),
                labels.data_ptr<int>() + n,
                inv_map.data_ptr<int>(),
                remapped_labels.data_ptr<int>());

            cudaDeviceSynchronize();

            return {centroids, remapped_labels};
        }

    } // namespace cuda
} // namespace gs
