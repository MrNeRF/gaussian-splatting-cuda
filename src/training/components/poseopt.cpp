#include <torch/torch.h>

namespace F = torch::nn::functional;

#include <core/poseopt.hpp>

// Converts a 6D rotation representation to a 3x3 rotation matrix
torch::Tensor rotation_6d_to_matrix(torch::Tensor rot_6d) {
    auto a1 = rot_6d.index({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3)});
    auto a2 = rot_6d.index({at::indexing::Ellipsis, at::indexing::Slice(3, at::indexing::None)});
    auto b1 = F::normalize(a1, F::NormalizeFuncOptions().dim(-1));
    auto b2 = a2 - (b1 * a2).sum(-1, true) * b1;
    b2 = F::normalize(b2, F::NormalizeFuncOptions().dim(-1));
    auto b3 = torch::cross(b1, b2, -1);
    return torch::stack({b1, b2, b3}, -2);
}

namespace gs {
    DirectPoseOptimizationModule::DirectPoseOptimizationModule(int number_of_cameras)
        : camera_embeddings(register_module("camera_embeddings",
                                            torch::nn::Embedding(number_of_cameras, 9))),
          rot_identity(register_buffer(
              "rot_identity",
              torch::tensor({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}))) {
        torch::nn::init::zeros_(camera_embeddings->weight);
    }
    // Only supports 1d batch size
    torch::Tensor DirectPoseOptimizationModule::forward(torch::Tensor camera_transforms, torch::Tensor embedding_ids) {
        auto bs = camera_transforms.size(0);
        auto delta_transformation = camera_embeddings(embedding_ids);
        auto delta_translation = delta_transformation.index({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3)});
        auto delta_rotation = delta_transformation.index({at::indexing::Ellipsis, at::indexing::Slice(3, at::indexing::None)});
        auto delta_rotation_matrix = rotation_6d_to_matrix(delta_rotation + rot_identity.expand({bs, -1}));

        auto transform = torch::eye(4, camera_transforms.options()).repeat({bs, 1, 1});
        transform.index_put_({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3), at::indexing::Slice(at::indexing::None, 3)},
                             delta_rotation_matrix);
        transform.index_put_({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3), 3},
                             delta_translation);
        return torch::matmul(camera_transforms, transform);
    }
    MLPPoseOptimizationModule::MLPPoseOptimizationModule(int number_of_cameras, int width, int depth) : camera_embeddings(register_module("camera_embeddings",
                                                                                                                                          torch::nn::Embedding(number_of_cameras, width))),
                                                                                                        rot_identity(register_buffer(
                                                                                                            "rot_identity",
                                                                                                            torch::tensor({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}))),
                                                                                                        mlp(register_module("mlp", torch::nn::Sequential())) {
        torch::nn::init::zeros_(camera_embeddings->weight);
        for (int i = 0; i < depth; ++i) {
            mlp->push_back(torch::nn::Linear(width, width));
            mlp->push_back(torch::nn::ReLU());
        }
        auto last_layer = torch::nn::Linear(width, 9);
        torch::nn::init::zeros_(last_layer->weight);
        torch::nn::init::zeros_(last_layer->bias);
        mlp->push_back(last_layer);
    }

    torch::Tensor MLPPoseOptimizationModule::forward(torch::Tensor camera_transforms, torch::Tensor embedding_ids) {
        auto bs = camera_transforms.size(0);
        auto camera_embedding = camera_embeddings(embedding_ids);
        auto delta_transformation = mlp->forward(camera_embedding);
        auto delta_translation = delta_transformation.index({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3)});
        auto delta_rotation = delta_transformation.index({at::indexing::Ellipsis, at::indexing::Slice(3, at::indexing::None)});
        auto delta_rotation_matrix = rotation_6d_to_matrix(delta_rotation + rot_identity.expand({bs, -1}));
        auto transform = torch::eye(4, camera_transforms.options()).repeat({bs, 1, 1});
        transform.index_put_({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3), at::indexing::Slice(at::indexing::None, 3)},
                             delta_rotation_matrix);
        transform.index_put_({at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, 3), 3},
                             delta_translation);
        return torch::matmul(camera_transforms, transform);
    }

} // namespace gs
