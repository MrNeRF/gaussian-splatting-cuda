#pragma once

#include "core/Dataset.hpp"
#include "core/point_cloud.hpp"
#include "adjacency_graph.h"

#include <torch/torch.h>
#include <tuple>

namespace densepcd {

  void get_dense_points(const std::shared_ptr<gs::CameraDataset> cameras, gs::PointCloud& pcd);

} // namespace densepcd
