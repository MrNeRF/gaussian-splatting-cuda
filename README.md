# "3D Gaussian Splatting for Real-Time Radiance Field Rendering" Reproduction in C++ and CUDA
This repository contains a reproduction of the Gaussian-Splatting software, originally developed by Inria and the Max Planck Institut for Informatik (MPII). The reproduction is written in C++ and CUDA.
I have used the source code from the original [repo](https://github.com/graphdeco-inria/gaussian-splatting) as blueprint for my first implementation. 
The original code is written in Python and PyTorch.

I embarked on this project to deepen my understanding of the groundbreaking paper on 3D Gaussian splatting, by reimplementing everything from scratch.

## About this Project
This project is a derivative of the original Gaussian-Splatting software and is governed by the Gaussian-Splatting License, which can be found in the LICENSE file in this repository. The original software was developed by Inria and MPII.

Please be advised that the software in this repository cannot be used for commercial purposes without explicit consent from the original licensors, Inria and MPII.

## Current Measurments as of 2023-08-17 
NVIDIA GeForce RTX 4090

    tandt/truck:
        ~100 seconds for 7000 iterations (my initial 2023-08-16) 
        ~120 seconds for 7000 iterations (my implementation 2023-08-16) 
        ~122 seconds for 7000 iterations (original PyTorch implementation)

NVIDIA GeForce RTX 3090

    tandt/truck:
        ~180 seconds for 7000 iterations (Latest 2023-08-17)
        ~200 seconds for 7000 iterations (2023-08-16)


While completely unoptimized, the gains in performance, though modest, are noteworthy.

=> Next Goal: Achieve 60 seconds for 7000 iterations in my implementation

## Build and Execution instructions
### Software Prerequisites 
1. Linux (tested with Ubuntu 22.04), windows probably won't work.
2. CMake 3.22 or higher.
3. CUDA 12.2 or higher (might work with less, has to be manually set and tested).
4. Python with development headers.
5. libtorch: You can find the setup instructions in the libtorch section of this README.
6. Other dependencies will be handled by the CMake script.

### Hardware Prerequisites
1. NVIDIA GPU with CUDA support (tested with RTX 4090 and RTX A5000) 

Not sure if it works with something smaller like RT 3080 Ti or similar hardware.

### Build
```bash
git clone --recursive https://github.com/MrNeRF/gaussian-splatting-cuda
cd gaussian-splatting-cuda
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip  
unzip  libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip -d external/
rm libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j
```

### Dataset
The dataset is not included in this repository. You can download it from the original repository under the following link:
[tanks & trains](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).
Then unzip it in the dataset folder.

### Running the program (training on the dataset)
```bash
./build/gaussian-splatting-cuda dataset/tandt/truck
```
The output is placed in the ouput folder in the root directory of this project.

### View the results
For now, you will need the SIBR view
```bash
git clone --recursive https://gitlab.inria.fr/sibr/sibr_core SIBR_core
cd SIBR_viewers
cmake -B build .
cmake --build build --target install --config Release -- -j 
cd ..
```

Then, you can view the results with:
```bash
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m output
```

## libtorch
Initially, I utilized libtorch to simplify the development process. Once the implementation is stable with libtorch, I will begin replacing torch elements with my custom CUDA implementation.
## MISC
Here is random collection of things that have to be described in README later on
- Needed for simple-knn: 
```bash sudo apt-get install python3-dev ```
 

## TODO (in no particular order, reminders for myself)
- [ ] Speed up with shifting stuff to CUDA.
- [ ] Need to think about the cameras. Separating camera and camera_info seems useless.
- [ ] Proper logging. (Lets see, low prio)
- [ ] Proper config file or cmd line config.

## Contributions
Contributions are welcome! I want to make this a community project. 

Some ideas for relative straight forward contributions:
- Revamp the README.
- Add a proper config file or cmd line config.

I want to get rid of some heavy dependencies:
- Replace glm with custom matrix operations
- Replace the few Eigen with some custom matrix operations

Advanced contributions:
- Build a renderer to view training output in real time and to replace SIBR viewer.
- Look into [gtsfm](https://github.com/borglab/gtsfm) to replace colmap dependency
- CUDA optimization
- ...

Own ideas are welcome as well!

## Citation and References
If you utilize this software or present results obtained using it, please reference the original work:

Kerbl, Bernhard; Kopanas, Georgios; Leimkühler, Thomas; Drettakis, George (2023). [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). ACM Transactions on Graphics, 42(4).

This will ensure the original authors receive the recognition they deserve.

## License

This project is licensed under the Gaussian-Splatting License - see the [LICENSE](LICENSE) file for details.

Follow me on Twitter if you want to know more about the latest development: https://twitter.com/janusch_patas