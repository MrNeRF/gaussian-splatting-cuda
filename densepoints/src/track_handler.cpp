#include "track_handler.h"

std::pair<torch::Tensor, torch::Tensor> densepcd::TrackHandler::get_point_cloud() {
    std::vector<torch::Tensor> pos;
    pos.reserve(tracks.size());
    std::vector<torch::Tensor> color;
    color.reserve(tracks.size());
    for (auto& [idx, track] : tracks) {
        pos.push_back(track.point3D.value());

        auto colorTensor = torch::from_blob(track.observations.at(0).rgb, {3}, torch::kU8);
        color.push_back(colorTensor);
    }
    return std::make_pair(torch::stack(pos), torch::stack(color).to(torch::kFloat));
}

void densepcd::TrackHandler::triangulateTrack(int tid) {
    auto& track = tracks[tid];
    if (track.observations.size() < 2)
        return; // need ≥ 2 views

    std::vector<torch::Tensor> A_rows;
    for (const auto& obs : track.observations) {
        if (!cameras.count(obs.camera_id))
            continue;
        auto P = cameras[obs.camera_id].P; // [3x4]

        // Build equations: u*P3 - P1, v*P3 - P2
        auto P1 = P.index({0});
        auto P2 = P.index({1});
        auto P3 = P.index({2});

        auto row1 = obs.u * P3 - P1;
        auto row2 = obs.v * P3 - P2;

        A_rows.push_back(row1);
        A_rows.push_back(row2);
    }

    if (A_rows.size() < 4)
        return; // insufficient constraints

    auto A = torch::stack(A_rows); // [2M x 4]

    // Solve min ||A*X||, subject to X[3]=1
    auto U = std::get<0>(A.svd());
    auto V = std::get<2>(A.svd());
    auto X = V.index({torch::indexing::Slice(), -1}); // last column
    X = X / X[-1];                                    // homogenize

    track.point3D = X.index({torch::indexing::Slice(0, 3)}).clone(); // store [3]
}