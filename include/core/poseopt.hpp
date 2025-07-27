#pragma once
#include "torch/nn/module.h"
#include "torch/nn/modules/embedding.h"

namespace gs {
    struct DirectPoseOptimizationModule : torch::nn::Module {
        DirectPoseOptimizationModule(int number_of_cameras);
        torch::Tensor forward(torch::Tensor camera_transforms, torch::Tensor embedding_ids);
        torch::nn::Embedding camera_embeddings; // [C, 9] for camera translation and 6D rotation
        torch::Tensor rot_identity; // [6] identity rotation in 6D representation
    };
}