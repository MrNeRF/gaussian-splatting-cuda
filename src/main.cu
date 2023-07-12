#include "camera.cuh"
#include "camera_utils.cuh"
#include "read_utils.cuh"
#include <filesystem>
#include <iostream>

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cout << "Usage: ./readPly <ply file>" << std::endl;
        return 1;
    }

    auto file_path = std::filesystem::path(argv[1]);

    read_ply_file(file_path / "sparse/0/points3D.ply");
    read_colmap_scene_info(file_path);

    auto cam = Camera(0);
    cam._camera_ID = 22;
    camera_to_JSON(cam);
    return 0;
}