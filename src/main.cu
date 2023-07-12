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

    return 0;
}