#pragma once
#include <torch/torch.h>

struct PointCloud;

namespace gauss::init {

    /// Groups every tensor required to bootstrap a GaussianModel.
    struct InitTensors {
        torch::Tensor xyz;
        torch::Tensor scaling;
        torch::Tensor rotation;
        torch::Tensor opacity;
        torch::Tensor features_dc;
        torch::Tensor features_rest;
        float scene_scale = -1.0;
    };

    /// Heavy-weight routine that used to live inside
    /// GaussianModel::Create_from_pcd.
    InitTensors build_from_point_cloud(PointCloud& pcd,
                                       int max_sh_degree,
                                       float scene_scale);

} // namespace gauss::init
