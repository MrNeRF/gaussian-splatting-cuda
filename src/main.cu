#include "camera.cuh"
#include "camera_utils.cuh"
#include "gaussian.cuh"
#include "loss_utils.cuh"
#include "parameters.cuh"
#include "read_utils.cuh"
#include "scene.cuh"
#include <iostream>
#include <torch/torch.h>

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cout << "Usage: ./readPly <ply file>" << std::endl;
        return 1;
    }
    auto t1 = torch::rand({2, 3});
    auto t2 = torch::rand({2, 3});
    gaussian_splatting::l1_loss(t1, t2);
    // TODO: read parameters from JSON file or command line
    auto modelParams = ModelParameters();
    modelParams.source_path = argv[1];
    const auto optimParams = OptimizationParameters();
    const auto pipelineParams = PipelineParameters();
    auto gaussians = GaussianModel(modelParams.sh_degree);
    auto scene = Scene(gaussians, modelParams);
    gaussians.Training_setup(optimParams);
    if (!torch::cuda::is_available()) {
        // At the moment, I want to make sure that my GPU is utilized.
        std::cout << "CUDA is not available! Training on CPU." << std::endl;
        exit(-1);
    }

    torch::Tensor bg_color = torch::tensor({1, 1, 1}).to(torch::kCUDA);
    for (int i = 0; i < optimParams.iterations; ++i) {
        if (i % 1000 == 0) {
            gaussians.One_up_sh_degree();
        }
    }
    //    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    //    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    //
    //    iter_start = torch.cuda.Event(enable_timing = True)
    //    iter_end = torch.cuda.Event(enable_timing = True)
    //
    //    viewpoint_stack = None
    //    ema_loss_for_log = 0.0
    //    for iteration in range(1, opt.iterations + 1):
    //        if network_gui.conn == None:
    //    {
    //        // compile test
    //        torch::Tensor tensor = torch::rand({2, 3});
    //        tensor.to(torch::kCUDA);
    //    }
    //    auto file_path = std::filesystem::path(argv[1]);
    //
    //    read_ply_file(file_path / "sparse/0/points3D.ply");
    //    read_colmap_scene_info(file_path);
    //
    //    auto cam = Camera(0);
    //    cam._camera_ID = 22;
    //    camera_to_JSON(cam);
    return 0;
}