#include "core/argument_parser.hpp"
#include "core/dataset.hpp"
#include "core/debug_utils.hpp"
#include "core/exporter.hpp"
#include "core/gaussian_init.hpp"
#include "core/inria_adc.hpp"
#include "core/parameters.hpp"
#include "core/render_utils.hpp"
#include "core/training_progress.hpp"
#include "kernels/fused_ssim.cuh"
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
    auto scene = dataset->get_scene_info();
    const std::size_t dataset_size = dataset->size().value();

    // Helper to create fresh dataloaders as needed
    const auto make_dataloader = [&](int workers = 4) {
        return create_dataloader_from_dataset(dataset, workers);
    };

    auto train_dataloader = make_dataloader();

    //----------------------------------------------------------------------
    // 4. Model initialisation
    //----------------------------------------------------------------------
    auto init = gauss::init::build_from_point_cloud(scene._point_cloud,
                                                    modelParams.sh_degree,
                                                    scene._nerf_norm_radius);

    auto strategy = InriaADC(modelParams.sh_degree, std::move(init));
    strategy.initialize(optimParams);

    auto background = modelParams.white_background
                          ? torch::tensor({1.f, 1.f, 1.f})
                          : torch::tensor({0.f, 0.f, 0.f},
                                          torch::TensorOptions().dtype(torch::kFloat32))
                                .to(torch::kCUDA);

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

            auto r_output = render(cam, strategy.get_model(), background);

            if (r_output.image.dim() == 3)
                r_output.image = r_output.image.unsqueeze(0); // NCHW for SSIM
            if (gt_image.dim() == 3)
                gt_image = gt_image.unsqueeze(0);

            if (r_output.image.sizes() != gt_image.sizes()) {
                std::cerr << "ERROR: size mismatch – rendered " << r_output.image.sizes()
                          << " vs. ground truth " << gt_image.sizes() << '\n';
                return -1;
            }

            //------------------------------------------------------------------
            // Loss = (1-λ)·L1 + λ·DSSIM
            //------------------------------------------------------------------
            auto l1l = torch::l1_loss(r_output.image.squeeze(0), gt_image.squeeze(0));
            auto ssim_loss = fused_ssim(r_output.image, gt_image, "same", /*train=*/true);
            auto loss = (1.f - optimParams.lambda_dssim) * l1l +
                        optimParams.lambda_dssim * (1.f - ssim_loss);
            loss.backward();
            const float loss_value = loss.item<float>();

            const bool is_densifying = (iter < optimParams.densify_until_iter &&
                                        iter > optimParams.densify_from_iter &&
                                        iter % optimParams.densification_interval == 0);
            //------------------------------------------------------------------
            // No-grad section – update radii, densify, optimise
            //------------------------------------------------------------------
            {
                torch::NoGradGuard no_grad;

                if (iter % 7000 == 0) {
                    auto pc = strategy.get_model().to_point_cloud();
                    write_ply(pc, modelParams.output_path, iter, /*join=*/false);
                }

                strategy.post_backward(iter, r_output);
                strategy.step(iter);
            }

            progress.update(iter, loss_value, static_cast<int>(strategy.get_model().size()), is_densifying);
            ++iter;
        }

        // Re-shuffle for the next epoch
        train_dataloader = make_dataloader();
    }

    auto pc = strategy.get_model().to_point_cloud();
    write_ply(pc, modelParams.output_path, iter, /*join=*/true);
    progress.print_final_summary(static_cast<int>(strategy.get_model().size()));
    return 0;
}