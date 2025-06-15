#include "Ops.h"
#include <gtest/gtest.h>
#include <iostream>
#include <torch/torch.h>

class IntersectDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        device = torch::kCUDA;
    }

    torch::Device device{torch::kCPU};
};

TEST_F(IntersectDebugTest, SingleCameraTest) {
    torch::manual_seed(42);

    // Test with single camera first
    int C = 1;
    int N = 10; // Small number for debugging
    int width = 64, height = 64;
    int tile_size = 16;

    auto means2d = torch::rand({C, N, 2}, device) * width;
    auto radii = torch::ones({C, N, 2}, torch::TensorOptions().dtype(torch::kInt32).device(device)) * 5;
    auto depths = torch::rand({C, N}, device);

    int tile_width = (width + tile_size - 1) / tile_size;
    int tile_height = (height + tile_size - 1) / tile_size;

    std::cout << "Input shapes:" << std::endl;
    std::cout << "  means2d: " << means2d.sizes() << std::endl;
    std::cout << "  radii: " << radii.sizes() << std::endl;
    std::cout << "  depths: " << depths.sizes() << std::endl;
    std::cout << "  tile_width: " << tile_width << ", tile_height: " << tile_height << std::endl;

    auto empty_orders = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto empty_tiles_per_gauss = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    auto [tiles_per_gauss, isect_ids, flatten_ids] = gsplat::intersect_tile(
        means2d, radii, depths,
        empty_orders,
        empty_tiles_per_gauss,
        C, tile_size, tile_width, tile_height,
        true // sort
    );

    std::cout << "\nOutput shapes:" << std::endl;
    std::cout << "  tiles_per_gauss: " << tiles_per_gauss.sizes() << " (dim=" << tiles_per_gauss.dim() << ")" << std::endl;
    std::cout << "  isect_ids: " << isect_ids.sizes() << " (dim=" << isect_ids.dim() << ")" << std::endl;
    std::cout << "  flatten_ids: " << flatten_ids.sizes() << " (dim=" << flatten_ids.dim() << ")" << std::endl;

    // Print some values for debugging
    if (tiles_per_gauss.numel() > 0 && tiles_per_gauss.numel() <= 20) {
        std::cout << "\ntiles_per_gauss values: " << tiles_per_gauss << std::endl;
    }
}

TEST_F(IntersectDebugTest, MultiCameraTest) {
    torch::manual_seed(42);

    // Test with multiple cameras
    int C = 3;
    int N = 5; // Very small for debugging
    int width = 32, height = 32;
    int tile_size = 16;

    auto means2d = torch::rand({C, N, 2}, device) * width;
    auto radii = torch::ones({C, N, 2}, torch::TensorOptions().dtype(torch::kInt32).device(device)) * 3;
    auto depths = torch::rand({C, N}, device);

    int tile_width = (width + tile_size - 1) / tile_size;
    int tile_height = (height + tile_size - 1) / tile_size;

    std::cout << "\nMulti-camera test:" << std::endl;
    std::cout << "Input shapes:" << std::endl;
    std::cout << "  means2d: " << means2d.sizes() << std::endl;
    std::cout << "  C=" << C << ", N=" << N << std::endl;

    auto empty_orders = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto empty_tiles_per_gauss = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    auto [tiles_per_gauss, isect_ids, flatten_ids] = gsplat::intersect_tile(
        means2d, radii, depths,
        empty_orders,
        empty_tiles_per_gauss,
        C, tile_size, tile_width, tile_height,
        true // sort
    );

    std::cout << "\nOutput shapes:" << std::endl;
    std::cout << "  tiles_per_gauss: " << tiles_per_gauss.sizes() << " (dim=" << tiles_per_gauss.dim() << ")" << std::endl;
    std::cout << "  isect_ids: " << isect_ids.sizes() << " (dim=" << isect_ids.dim() << ")" << std::endl;
    std::cout << "  flatten_ids: " << flatten_ids.sizes() << " (dim=" << flatten_ids.dim() << ")" << std::endl;

    // Check if tiles_per_gauss shape matches expectations
    if (tiles_per_gauss.dim() == 2) {
        std::cout << "tiles_per_gauss is 2D with shape [" << tiles_per_gauss.size(0) << ", " << tiles_per_gauss.size(1) << "]" << std::endl;
        EXPECT_EQ(tiles_per_gauss.size(0), C);
        EXPECT_EQ(tiles_per_gauss.size(1), N);
    } else if (tiles_per_gauss.dim() == 1) {
        std::cout << "tiles_per_gauss is 1D with length " << tiles_per_gauss.size(0) << std::endl;
        EXPECT_EQ(tiles_per_gauss.size(0), C * N);
    }
}
