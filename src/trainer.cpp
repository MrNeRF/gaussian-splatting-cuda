#include "core/trainer.hpp"
#include "core/render_utils.hpp"
#include "kernels/fused_ssim.cuh"
#include <c10/cuda/CUDACachingAllocator.h>
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

        // Check CUDA availability
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        // Initialize strategy with optimization parameters
        strategy_->initialize(params.optimization);

        // Set up background color
        background_ = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32));
        background_ = background_.to(torch::kCUDA);

        // Initialize progress tracking
        progress_ = std::make_unique<TrainingProgress>(
            params.optimization.iterations,
            /*bar_width=*/100);
    }

    auto Trainer::make_dataloader(int workers) const {
        return create_dataloader_from_dataset(dataset_, workers);
    }

    void Trainer::train() {
        int iter = 1;
        int epochs_needed = (params_.optimization.iterations + dataset_size_ - 1) / dataset_size_;

        // Create initial dataloader
        auto train_dataloader = make_dataloader();

        for (int epoch = 0; epoch < epochs_needed && iter <= params_.optimization.iterations; ++epoch) {
            for (auto& batch : *train_dataloader) { // batch = std::vector<CameraExample>
                if (iter > params_.optimization.iterations) {
                    break;
                }

                auto& example = batch[0];
                Camera cam = std::move(example.data); // <-- no ()

                // Initialize CUDA tensors in the main thread
                cam.initialize_cuda_tensors();

                auto gt_image = cam.Get_original_image().to(torch::kCUDA, /*non_blocking=*/true);
                auto r_output = render_with_gsplat(cam, strategy_->get_model(), background_);

                if (r_output.image.dim() == 3)
                    r_output.image = r_output.image.unsqueeze(0); // NCHW for SSIM
                if (gt_image.dim() == 3)
                    gt_image = gt_image.unsqueeze(0);

                if (r_output.image.sizes() != gt_image.sizes()) {
                    std::cerr << "ERROR: size mismatch – rendered " << r_output.image.sizes()
                              << " vs. ground truth " << gt_image.sizes() << '\n';
                    throw std::runtime_error("Image size mismatch");
                }

                //------------------------------------------------------------------
                // Loss = (1-λ)·L1 + λ·DSSIM
                //------------------------------------------------------------------
                auto l1l = torch::l1_loss(r_output.image.squeeze(0), gt_image.squeeze(0));
                auto ssim_loss = fused_ssim(r_output.image, gt_image, "same", /*train=*/true);
                auto loss = (1.f - params_.optimization.lambda_dssim) * l1l +
                            params_.optimization.lambda_dssim * (1.f - ssim_loss);
                loss.backward();
                const float loss_value = loss.item<float>();

                const bool is_densifying = (iter < params_.optimization.densify_until_iter &&
                                            iter > params_.optimization.densify_from_iter &&
                                            iter % params_.optimization.densification_interval == 0);

                //------------------------------------------------------------------
                // No-grad section – update radii, densify, optimise
                //------------------------------------------------------------------
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

            // Re-shuffle for the next epoch
            train_dataloader = make_dataloader();
        }

        // Save final model
        strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
        progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
    }

} // namespace gs
