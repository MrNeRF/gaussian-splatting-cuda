#include "dense_point_cloud.h"

void densepcd::get_dense_points(const std::shared_ptr<gs::CameraDataset> cameras, gs::PointCloud& pcd) {
    printf("Starting dense point cloud creation\n");
	
	// Find reasonable far plane estimate from camera positions
	
	// Build adj graph
    AdjacencyGraph graph;
    for (size_t idx = 0; idx < cameras->size(); idx++) {
        auto camEx = cameras->get(idx);
        graph.add_camera(camEx.data);
	}
	graph.compute_adjancency();

	// Match cameras = build track handler

	// Get dense pcd from track handler
}
