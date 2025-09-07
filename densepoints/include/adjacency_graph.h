#pragma once

#include "core/Dataset.hpp"
#include <map>
#include <set>
#include <utility>

namespace densepcd {
    class AdjacencyGraph {
    private:
        struct Node {
            gs::CameraWithImage camImage;
            std::unordered_set<int> adj; // idx 
            
            std::vector<torch::Tensor> frustumCorners;

            Node(gs::CameraWithImage camWithImage) { camImage = camWithImage; }
            Node() = default;
        };

        std::unordered_map<int, Node> _graph;
        std::vector<int> _idxs;

        float _nearPlane = 0.1f;
        float _farPlane = 10.f;

        bool frustums_overlap(const Node& node1, const Node& node2);
    public:     

        AdjacencyGraph(float nearPlane = 0.1f, float farPlane = 10.f) : _nearPlane(nearPlane), _farPlane(farPlane){
            
        }

        void add_camera(gs::CameraWithImage camWithImage);

        void compute_adjancency(int n_nearest_neighbors = 5);
    };

    bool point_in_frustum(const gs::Camera& cam, torch::Tensor X, float nearPlane, float farPlane);

    std::vector<torch::Tensor> frustum_corners(const gs::Camera& cam, float nearPlane, float farPlane);

} // namespace densepcd