#include "core/argument_parser.hpp"
#include "core/debug_utils.hpp"
#include "core/gaussian.hpp"
#include "core/parameters.hpp"
#include "core/render_utils.hpp"
#include "core/dataset.hpp"
#include "core/training_progress.hpp"
#include "kernels/fused_ssim.cuh"
#include "kernels/loss_utils.cuh"
#include <c10/cuda/CUDACachingAllocator.h>
#include <iostream>
#include <torch/torch.h>

int main(int argc, char* argv[]) {
    auto args = gs::args::convert_args(argc, argv);
    auto modelParams = gs::param::ModelParameters();
    auto optimParams = gs::param::read_optim_params_from_json();

    if (gs::args::parse_arguments(args, modelParams, optimParams) < 0) {
        return -1;
    }

    // Create DataLoader with 4 worker threads (like PyTorch)
    auto dataloader = create_dataloader(modelParams,
                                        /*num_workers=*/4,
                                        /*pin_memory=*/true);

    // Initialize Gaussians
    auto gaussians = GaussianModel(modelParams.sh_degree);
    const auto& scene_info = dataloader->get_scene_info();

    // Fix const issue - make a copy for Create_from_pcd
    PointCloud point_cloud_copy = scene_info._point_cloud;
    gaussians.Create_from_pcd(point_cloud_copy, scene_info._nerf_norm_radius);
    gaussians.Training_setup(optimParams);

    if (!torch::cuda::is_available()) {
        std::cout << "CUDA is not available! Training on CPU." << std::endl;
        exit(-1);
    }

    auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    auto background = modelParams.white_background ?
                                                   torch::tensor({1.f, 1.f, 1.f}) :
                                                   torch::tensor({0.f, 0.f, 0.f}, pointType).to(torch::kCUDA);

    // Store cameras extent for use in densification
    float cameras_extent = scene_info._nerf_norm_radius;

    TrainingProgress progress(optimParams.iterations, 100);

    // Training loop with parallel data loading!
    std::cout << "Starting training with parallel data loading..." << std::endl;

    for (int iter = 1; iter < optimParams.iterations + 1; ++iter) {
        // Get next camera - loaded in parallel by worker threads!
        Camera cam = dataloader->get_next_camera();
        auto gt_image = cam.Get_original_image().to(torch::kCUDA, true);

        if (iter % 1000 == 0) {
            gaussians.One_up_sh_degree();
        }

        // Render
        auto [image, viewspace_point_tensor, visibility_filter, radii] = render(cam, gaussians, background);

        // Ensure both images are 4D tensors [N, C, H, W] for SSIM
        if (image.dim() == 3) {
            image = image.unsqueeze(0);
        }
        if (gt_image.dim() == 3) {
            gt_image = gt_image.unsqueeze(0);
        }

        // Verify tensor shapes
        if (image.sizes() != gt_image.sizes()) {
            std::cerr << "ERROR: Image size mismatch - rendered: " << image.sizes()
                      << ", ground truth: " << gt_image.sizes() << std::endl;
            exit(-1);
        }

        // Loss Computations
        auto image_for_l1 = image.dim() == 4 && image.size(0) == 1 ? image.squeeze(0) : image;
        auto gt_for_l1 = gt_image.dim() == 4 && gt_image.size(0) == 1 ? gt_image.squeeze(0) : gt_image;

        auto l1l = gaussian_splatting::l1_loss(image_for_l1, gt_for_l1);
        auto ssim_loss = fused_ssim(image, gt_image, /*padding=*/"same", /*train=*/true);
        auto loss = (1.f - optimParams.lambda_dssim) * l1l + optimParams.lambda_dssim * (1.f - ssim_loss);

        loss.backward();

        {
            torch::NoGradGuard no_grad;
            auto visible_max_radii = gaussians._max_radii2D.masked_select(visibility_filter);
            auto visible_radii = radii.masked_select(visibility_filter);
            auto max_radii = torch::max(visible_max_radii, visible_radii);
            gaussians._max_radii2D.masked_scatter_(visibility_filter, max_radii);

            // Check for densification to show (+) indicator
            bool is_densifying = (iter < optimParams.densify_until_iter &&
                                  iter > optimParams.densify_from_iter &&
                                  iter % optimParams.densification_interval == 0);

            progress.update(iter, loss.item<float>(), static_cast<int>(gaussians.Get_xyz().size(0)), 0.0f, is_densifying);

            if (iter == optimParams.iterations) {
                gaussians.Save_ply(modelParams.output_path, iter, true);
                break;
            }

            if (iter % 7'000 == 0) {
                gaussians.Save_ply(modelParams.output_path, iter, false);
            }

            // Densification - Using cameras_extent instead of scene.Get_cameras_extent()
            if (iter < optimParams.densify_until_iter) {
                gaussians.Add_densification_stats(viewspace_point_tensor, visibility_filter);
                if (is_densifying) {
                    gaussians.Densify_and_prune(optimParams.densify_grad_threshold, optimParams.min_opacity, cameras_extent);
                }

                if (iter % optimParams.opacity_reset_interval == 0 || (modelParams.white_background && iter == optimParams.densify_from_iter)) {
                    gaussians.Reset_opacity();
                }
            }

            // Optimizer step
            if (iter < optimParams.iterations) {
                gaussians._optimizer->step();
                gaussians._optimizer->zero_grad(true);
                gaussians.Update_learning_rate(iter);
            }
        }
    }

    progress.print_final_summary(static_cast<int>(gaussians.Get_xyz().size(0)));
    return 0;
}