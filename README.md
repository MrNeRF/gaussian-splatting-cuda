# "3D Gaussian Splatting for Real-Time Radiance Field Rendering" Reproduction in C++ and CUDA
This repository contains a reproduction of the Gaussian-Splatting software, originally developed by Inria and the Max Planck Institut for Informatik (MPII). The reproduction is written in C++ and CUDA.

This is my attempt to learn more about the wonderful paper 3D gaussian splatting by reimplementing everthing from scratch. 
It is work in progress. I expect it to work in mid August 2023.
## About this Project

This project is a derivative work of the original Gaussian-Splatting software. It is governed by the Gaussian-Splatting License, which can be found in the `LICENSE` file in this repository. The original software was developed by Inria and MPII.

Please note that the software in this repository cannot be used for commercial purposes without explicit consent from the original licensors, Inria and MPII.

## libtorch
In the beginning I use libtorch to make my life easier. When everything works with libtorch, I will start replacing 
torch elements with my own custom cuda implementation.


Download the libtorch library using the following command:

```bash
wget https://download.pytorch.org/libtorch/test/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip  
```

This will download a zip file named `libtorch-shared-with-deps-latest.zip`. To extract this zip file, use the command:

```bash
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip -d external/
rm libtorch-cxx11-abi-shared-with-deps-latest.zip
```
This will create a folder named `libtorch` in the `external` directory of your project.

## MISC
Here is random collection of things that have to be described in README later on
- Needed for simple-knn: sudo apt-get install python3-dev 
- We need to patch pybind in libtorch on linux because nvcc has problems with a template definition
- git submodule update --init --recursive for glm in diff-gaussian-rasterization 

## Citation and References

When using this software, or presenting results obtained by it, please cite the original work as follows:

Kerbl, Bernhard; Kopanas, Georgios; Leimkühler, Thomas; Drettakis, George (2023). [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). ACM Transactions on Graphics, 42(4).

This ensures that the original authors receive the proper recognition for their work.

## License

This project is licensed under the Gaussian-Splatting License - see the [LICENSE](LICENSE) file for details.

