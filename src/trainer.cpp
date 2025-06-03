#include "core/trainer.hpp"
#include "core/image_io.hpp"
#include "core/rasterizer.hpp"
#include "kernels/fused_ssim.cuh"
#include <iostream>
#include <torch/torch.h>

namespace gs {

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     const param::TrainingParameters& params)
        : dataset_(std::move(dataset)),
          strategy_(std::move(strategy)),
          params_(params),
          dataset_size_(dataset_->size().value()) {

        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        strategy_->initialize(params.optimization);

        background_ = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32));
        background_ = background_.to(torch::kCUDA);

        progress_ = std::make_unique<TrainingProgress>(
            params.optimization.iterations,
            /*bar_width=*/100);
    }

    auto Trainer::make_dataloader(int workers) const {
        return create_dataloader_from_dataset(dataset_, workers);
    }

    // In trainer.cpp, update the train() method:
    void Trainer::train() {
        int iter = 1;
        int epochs_needed = (params_.optimization.iterations + dataset_size_ - 1) / dataset_size_;

        auto train_dataloader = make_dataloader();

        for (int epoch = 0; epoch < epochs_needed && iter <= params_.optimization.iterations; ++epoch) {
            for (auto& batch : *train_dataloader) {
                if (iter > params_.optimization.iterations) {
                    break;
                }

                auto camera_with_image = batch[0].data;
                Camera* cam = camera_with_image.camera;
                torch::Tensor gt_image = std::move(camera_with_image.image);

                auto r_output = gs::rasterize(*cam, strategy_->get_model(), background_, 1, false);

                // if (iter % 100 == 0) { // Save every 100 iterations
                //     auto save_path = params_.dataset.output_path /
                //                      ("render_iter_" + std::to_string(iter) + ".png");
                //     save_image(save_path, {gt_image, r_output.image}, true, 2);
                // }

                if (r_output.image.dim() == 3)
                    r_output.image = r_output.image.unsqueeze(0);

                if (gt_image.dim() == 3)
                    gt_image = gt_image.unsqueeze(0);

                if (r_output.image.sizes() != gt_image.sizes()) {
                    std::cerr << "ERROR: size mismatch – rendered " << r_output.image.sizes()
                              << " vs. ground truth " << gt_image.sizes() << '\n';
                    throw std::runtime_error("Image size mismatch");
                }

                // Base loss computation
                auto l1l = torch::l1_loss(r_output.image.squeeze(0), gt_image.squeeze(0));

                auto ssim_loss = fused_ssim(r_output.image, gt_image, "same", /*train=*/true);
                auto loss = (1.f - params_.optimization.lambda_dssim) * l1l +
                            params_.optimization.lambda_dssim * (1.f - ssim_loss);

                // Add opacity regularization
                if (params_.optimization.opacity_reg > 0.0f) {
                    auto opacity_l1 = torch::abs(strategy_->get_model().get_opacity()).mean();
                    loss = loss + params_.optimization.opacity_reg * opacity_l1;
                }

                // Add scale regularization
                if (params_.optimization.scale_reg > 0.0f) {
                    auto scale_l1 = torch::abs(strategy_->get_model().get_scaling()).mean();
                    loss = loss + params_.optimization.scale_reg * scale_l1;
                }

                loss.backward();
                const float loss_value = loss.item<float>();

                const bool is_densifying = (iter < params_.optimization.densify_until_iter &&
                                            iter > params_.optimization.densify_from_iter &&
                                            iter % params_.optimization.densification_interval == 0);

                {
                    torch::NoGradGuard no_grad;

                    if (iter % 7000 == 0) {
                        strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/false);
                    }

                    strategy_->post_backward(iter, r_output);
                    strategy_->step(iter);
                }

                progress_->update(iter, loss_value, static_cast<int>(strategy_->get_model().size()), is_densifying);
                ++iter;
            }

            train_dataloader = make_dataloader();
        }

        strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
        progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
    }

} // namespace gs