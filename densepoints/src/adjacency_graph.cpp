#include "adjacency_graph.h"

// Compute the full adjacency graph. Consider the n nearest neighbors based on camera position.
void densepcd::AdjacencyGraph::compute_adjancency(int n_nearest_neighbors) {
    for (int i = 0; i < _idxs.size() - 1; i++) {
        int idx1 = _idxs[i];
        auto& node1 = _graph[idx1];

        std::vector<std::pair<float, int>> distNeighbors;
        distNeighbors.reserve(_idxs.size() - i);
        for (int j = i + 1; j < _idxs.size(); j++) {
            int idx2 = _idxs[j];
            auto& node2 = _graph[idx2];            
            auto sqDist = torch::pow(node1.camImage.camera->cam_position() - node2.camImage.camera->cam_position(), 2).sum().item<float>(); // Can use sq dist instead of dist
            distNeighbors.emplace_back(sqDist, idx2);
        }

        std::sort(distNeighbors.begin(), distNeighbors.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first < b.first;
        });
        for (int j = 0; j < n_nearest_neighbors; j++) {
            auto& node2 = _graph[distNeighbors[j].second];
            auto overlaps = frustums_overlap(node1, node2);
            if (overlaps) {
                node1.adj.insert(node2.camImage.camera->uid());
                node2.adj.insert(node1.camImage.camera->uid());
            }
        }
    }

    auto node1 = _graph[_idxs[0]];
    printf("%s\n", node1.camImage.camera->image_name());
    for (auto& n : node1.adj) {
        printf("%s\n", _graph[n].camImage.camera->image_name());
    }
}

void densepcd::AdjacencyGraph::add_camera(gs::CameraWithImage camWithImage) {
    auto idx = camWithImage.camera->uid();
    _graph[idx] = Node(camWithImage);
    _idxs.push_back(idx);

    auto& node = _graph[idx];
    node.frustumCorners = frustum_corners(*node.camImage.camera, _nearPlane, _farPlane);
}

bool densepcd::point_in_frustum(const gs::Camera& cam, torch::Tensor X, float nearPlane, float farPlane) {
    torch::Tensor Xc = cam.R().matmul(X) + cam.T(); // cam coords [3,1]

    auto z = Xc[2].item<float>();
    if (z < nearPlane || z > farPlane)
        return false;

    auto [fx, fy, cx, cy] = cam.get_intrinsics();

    auto u = fx * (Xc[0].item<float>() / z) + cx;
    auto v = fy * (Xc[1].item<float>() / z) + cy;

    return (u >= 0 && u < cam.image_width() && v >= 0 && v < cam.image_height());
}

bool densepcd::AdjacencyGraph::frustums_overlap(const Node& node1, const Node& node2) {
    for (auto& c : node1.frustumCorners)
        if (point_in_frustum(*node2.camImage.camera, c, _nearPlane, _farPlane))
            return true;

    for (auto& c : node2.frustumCorners)
        if (point_in_frustum(*node1.camImage.camera, c, _nearPlane, _farPlane))
            return true;

    return false;
}

std::vector<torch::Tensor> densepcd::frustum_corners(const gs::Camera& cam, float nearPlane, float farPlane) {
    float pixels[8] = {
        0, 0,
        cam.image_width(), 0,
        0, cam.image_height(),
        cam.image_width(), cam.image_height()};

    auto [fx, fy, cx, cy] = cam.get_intrinsics();

    // Convert to rays and scale to depths
    auto RTrans = cam.R().transpose(0, 1);
    auto C = cam.cam_position().to(RTrans.device()); // For some reason cam_position is cuda, but R is not

    std::vector<torch::Tensor> corners;
    corners.reserve(8);
    for (auto depth : {nearPlane, farPlane}) {
        for (int i = 0; i < 4; i++) {
            float u = pixels[i * 2];
            float v = pixels[i * 2 + 1];

            float x = (u - cx) / fx;
            float y = (v - cy) / fy;

            auto ray = torch::tensor({x, y, 1.0f}).to(cam.R().device());
            ray /= ray.norm();
            ray *= depth;

            auto point = RTrans.matmul(ray) + C;
            corners.push_back(point);
        }
    }
    return corners;
}
