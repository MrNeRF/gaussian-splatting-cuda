#include "read_utils.cuh"
#include <iostream>

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cout << "Usage: ./readPly <ply file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    read_ply_file(filename);

    return 0;
}