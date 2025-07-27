#pragma once
#include "torch/nn/module.h"
#include "torch/nn/modules/embedding.h"

namespace gs {
    struct PoseOptimizationModule : torch::nn::Module {
        PoseOptimizationModule() {}
        virtual torch::Tensor forward(torch::Tensor camera_transforms, torch::Tensor embedding_ids) {
            // No operation, just return the input transforms
            return camera_transforms;
        }
    };
    struct DirectPoseOptimizationModule : PoseOptimizationModule {
        explicit DirectPoseOptimizationModule(int number_of_cameras);
        torch::Tensor forward(torch::Tensor camera_transforms, torch::Tensor embedding_ids) override;
        torch::nn::Embedding camera_embeddings; // [C, 9] for camera translation and 6D rotation
        torch::Tensor rot_identity; // [6] identity rotation in 6D representation
    };
    struct MLPPoseOptimizationModule : PoseOptimizationModule {
        explicit MLPPoseOptimizationModule(int number_of_cameras, int width = 64, int depth = 2);
        torch::Tensor forward(torch::Tensor camera_transforms, torch::Tensor embedding_ids) override;
        torch::nn::Embedding camera_embeddings; // [C, F]
        torch::Tensor rot_identity; // [6] identity rotation in 6D representation
        torch::nn::Sequential mlp;
    };
}