#include "dense_point_cloud.h"
#include "matcher.h"

void densepcd::get_dense_points(const std::shared_ptr<gs::CameraDataset> cameras, gs::PointCloud& pcd) {
    using namespace torch::indexing;

    printf("Starting dense point cloud creation\n");
	
	// Find reasonable far plane estimate from camera positions
	
	// Build adj graph
    AdjacencyGraph graph;
    for (size_t idx = 0; idx < cameras->size(); idx++) {
        auto camEx = cameras->get(idx);
        graph.add_camera(camEx.data);
	}
	graph.compute_adjancency(20);

    bool useLoftr = true;
    if (!useLoftr) {
        XFeatFeatureExtractor featureExtractor(1414, 1061, 6144);
        // Extract features from all cameras
        for (auto& [idx, node] : graph.graph()) {
            auto [kpts, desc] = featureExtractor.extract_features(node.camImage.image);
            node.kpts = kpts;
            node.desc = desc;
        }
    }

    LOFTRMatcher loftrMatcher;

	// Match features and insert into track handler
    TrackHandler trackHandler;
    printf("Starting matching\n");
    std::unordered_map<int, int> matched; // Make sure you haven't matched this pair yet. Bad solution but it's almost 1 am.
    for (auto& [idx1, node1] : graph.graph()) {
		for (auto& idx2 : node1.adj) {
            auto lowKey = std::min(idx1, idx2);
            auto highKey = std::max(idx2, idx1);
			if (matched.find(lowKey) == matched.end()) {
                matched[lowKey] = highKey;
                auto node2 = graph.graph()[idx2];

                std::vector<std::pair<size_t, size_t>> matchesVec;
                if (!useLoftr) {
                    auto [feat_idx1, feat_idx2, mask] = match(node1.desc, node2.desc);
                    feat_idx1 = feat_idx1.index({mask});
                    feat_idx2 = feat_idx2.index({mask});

                    auto matches = torch::cat({feat_idx1.unsqueeze(-1), feat_idx2.unsqueeze(-1)}, 1);
                    matches = matches.to(torch::kU64).to(torch::kCPU);

                    matchesVec = std::vector<std::pair<size_t, size_t>>(matches.size(0));
                    memcpy(matchesVec.data(), matches.data_ptr(), matches.numel() * 8);
                } else {
                    auto [kpts1, kpts2] = loftrMatcher.matchImages(node1.camImage.image, node2.camImage.image);
                    auto matches = torch::arange(kpts1.size(0)).to(torch::kU64).to(torch::kCPU);
                    
                    node1.kpts = kpts1;
                    node2.kpts = kpts2;

                    matchesVec = std::vector<std::pair<size_t, size_t>>(matches.size(0));
                    memcpy(matchesVec.data(), matches.data_ptr(), matches.numel() * 8);
                }

                auto Rt1 = torch::hstack({node1.camImage.camera->R(), node1.camImage.camera->T().unsqueeze(1)}).to(torch::kCUDA);
                auto P1 = torch::matmul(node1.camImage.camera->K(), Rt1);

                auto Rt2 = torch::hstack({node2.camImage.camera->R(), node2.camImage.camera->T().unsqueeze(1)}).to(torch::kCUDA);
                auto P2 = torch::matmul(node2.camImage.camera->K(), Rt2);

                trackHandler.addCamera(idx1, P1.squeeze(0));
                trackHandler.addCamera(idx2, P2.squeeze(0));

                //if (trackHandler.numCameras() > 30)
                //    break;

                auto kpts1 = node1.kpts.to(torch::kCPU).data_ptr<float>();
                auto kpts2 = node2.kpts.to(torch::kCPU).data_ptr<float>();
                for (auto& match : matchesVec) {
                    auto u1 = kpts1[match.first * 2];
                    auto v1 = kpts1[match.first * 2 + 1];
                    auto u2 = kpts2[match.second * 2];
                    auto v2 = kpts2[match.second * 2 + 1];

                    if (u1 < 0 || u1 >= node1.camImage.image.size(2) || v1 < 0 || v1 >= node1.camImage.image.size(1))
                        continue;
                    if (u2 < 0 || u2 >= node2.camImage.image.size(2) || v2 < 0 || v2 >= node2.camImage.image.size(1))
                        continue;

                    unsigned char rgb1[3];
                    for (int i = 0; i < 3; i++)
                        rgb1[i] = (unsigned char)(node1.camImage.image.index({i, (int)v1, (int)u1}).item<float>() * 255);

                    unsigned char rgb2[3];
                    for (int i = 0; i < 3; i++)
                        rgb2[i] = (unsigned char)(node2.camImage.image.index({i, (int)v1, (int)u1}).item<float>() * 255);
                    trackHandler.insertMatch(idx1, match.first, u1, v1, rgb1, idx2, match.second, u2, v2, rgb2);
                }
			}
		}
    }
    printf("Starting triangulation\n");
	// Get dense pcd from track handler
    trackHandler.triangulate();

    auto [positions, colors] = trackHandler.get_point_cloud();

    pcd.means = positions.to(torch::kCUDA);
    pcd.colors = colors.to(torch::kCUDA);
    printf("Dense Point cloud done. Num Points: %d\n", pcd.means.size(0));
}
