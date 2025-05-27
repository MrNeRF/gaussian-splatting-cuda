#include "core/argument_parser.hpp"
#include "core/dataset.hpp"
#include "core/debug_utils.hpp"
#include "core/gaussian.hpp"
#include "core/parameters.hpp"
#include "core/render_utils.hpp"
#include "core/training_progress.hpp"
#include "kernels/fused_ssim.cuh"
#include "kernels/loss_utils.cuh"
#include <c10/cuda/CUDACachingAllocator.h>
#include <iostream>
#include <torch/torch.h>

int main(int argc, char* argv[]) {
    //----------------------------------------------------------------------
    // 1. Parse command-line arguments
    //----------------------------------------------------------------------
    auto args = gs::args::convert_args(argc, argv);
    auto modelParams = gs::param::ModelParameters();
    auto optimParams = gs::param::read_optim_params_from_json();
    if (gs::args::parse_arguments(args, modelParams, optimParams) < 0)
        return -1;

    //----------------------------------------------------------------------
    // 2. Initialize CUDA
    //----------------------------------------------------------------------
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available – aborting.\n";
        return -1;
    }

    //----------------------------------------------------------------------
    // 3. Create dataset
    //----------------------------------------------------------------------
    auto dataset = create_camera_dataset(modelParams);
    const auto& scene = dataset->get_scene_info();
    const std::size_t dataset_size = dataset->size().value();

    // Helper to create fresh dataloaders as needed
    const auto make_dataloader = [&](int workers = 4) {
        return create_dataloader_from_dataset(dataset, workers);
    };

    auto train_dataloader = make_dataloader();

    //----------------------------------------------------------------------
    // 4. Model initialisation
    //----------------------------------------------------------------------
    auto gaussians = GaussianModel(modelParams.sh_degree);

    PointCloud point_cloud_copy = scene._point_cloud;
    gaussians.Create_from_pcd(point_cloud_copy, scene._nerf_norm_radius);
    gaussians.Training_setup(optimParams);

    auto background = modelParams.white_background
                          ? torch::tensor({1.f, 1.f, 1.f})
                          : torch::tensor({0.f, 0.f, 0.f},
                                          torch::TensorOptions().dtype(torch::kFloat32))
                                .to(torch::kCUDA);

    const float cameras_extent = scene._nerf_norm_radius;
    TrainingProgress progress(optimParams.iterations, /*bar_width=*/100);

    int iter = 1;
    int epochs_needed = (optimParams.iterations + dataset_size - 1) / dataset_size;

    for (int epoch = 0; epoch < epochs_needed && iter <= optimParams.iterations; ++epoch) {
        for (auto& batch : *train_dataloader) { // batch = std::vector<CameraExample>
            if (iter > optimParams.iterations)
                break;

            auto& example = batch[0];
            Camera cam = std::move(example.data); // <-- no ()

            // Initialize CUDA tensors in the main thread
            cam.initialize_cuda_tensors();

            auto gt_image = cam.Get_original_image().to(torch::kCUDA, /*non_blocking=*/true);

            if (iter % 1000 == 0)
                gaussians.One_up_sh_degree();

            auto [image, viewspace_point_tensor, visibility_filter, radii] =
                render(cam, gaussians, background);

            if (image.dim() == 3)
                image = image.unsqueeze(0); // NCHW for SSIM
            if (gt_image.dim() == 3)
                gt_image = gt_image.unsqueeze(0);

            if (image.sizes() != gt_image.sizes()) {
                std::cerr << "ERROR: size mismatch – rendered " << image.sizes()
                          << " vs. ground truth " << gt_image.sizes() << '\n';
                return -1;
            }

            //------------------------------------------------------------------
            // Loss = (1-λ)·L1 + λ·DSSIM
            //------------------------------------------------------------------
            auto l1l = gaussian_splatting::l1_loss(image.squeeze(0), gt_image.squeeze(0));
            auto ssim_loss = fused_ssim(image, gt_image, "same", /*train=*/true);
            auto loss = (1.f - optimParams.lambda_dssim) * l1l +
                        optimParams.lambda_dssim * (1.f - ssim_loss);
            loss.backward();

            //------------------------------------------------------------------
            // No-grad section – update radii, densify, optimise
            //------------------------------------------------------------------
            {
                torch::NoGradGuard no_grad;

                auto visible_max_radii = gaussians._max_radii2D.masked_select(visibility_filter);
                auto visible_radii = radii.masked_select(visibility_filter);
                gaussians._max_radii2D.masked_scatter_(
                    visibility_filter, torch::max(visible_max_radii, visible_radii));

                bool is_densifying = (iter < optimParams.densify_until_iter &&
                                      iter > optimParams.densify_from_iter &&
                                      iter % optimParams.densification_interval == 0);

                float loss_value = loss.item<float>();
                progress.update(iter, loss_value,
                                static_cast<int>(gaussians.Get_xyz().size(0)),
                                /*psnr=*/0.0f, is_densifying);

                if (iter == optimParams.iterations) {
                    gaussians.Save_ply(modelParams.output_path, iter, /*final=*/true);
                    break;
                }
                if (iter % 7000 == 0)
                    gaussians.Save_ply(modelParams.output_path, iter, /*final=*/false);

                // Densification & pruning
                if (iter < optimParams.densify_until_iter) {
                    gaussians.Add_densification_stats(viewspace_point_tensor, visibility_filter);
                    if (is_densifying) {
                        gaussians.Densify_and_prune(optimParams.densify_grad_threshold,
                                                    optimParams.min_opacity,
                                                    cameras_extent);
                    }
                    if (iter % optimParams.opacity_reset_interval == 0 ||
                        (modelParams.white_background && iter == optimParams.densify_from_iter)) {
                        gaussians.Reset_opacity();
                    }
                }

                // Optimiser step
                if (iter < optimParams.iterations) {
                    gaussians._optimizer->step();
                    gaussians._optimizer->zero_grad(true);
                    gaussians.Update_learning_rate(iter);
                }
            }

            ++iter;
        }

        // Re-shuffle for the next epoch
        train_dataloader = make_dataloader();
    }

    progress.print_final_summary(static_cast<int>(gaussians.Get_xyz().size(0)));
    return 0;
}