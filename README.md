# "3D Gaussian Splatting for Real-Time Radiance Field Rendering" Reproduction in C++ and CUDA
This repository contains a reproduction of the Gaussian-Splatting software, originally developed by Inria and the Max Planck Institut for Informatik (MPII). The reproduction is written in C++ and CUDA.
I have used the source code from the original [repo](https://github.com/graphdeco-inria/gaussian-splatting) as blueprint for my first implementation.
The original code is written in Python and PyTorch.

I embarked on this project to deepen my understanding of the groundbreaking paper on 3D Gaussian splatting, by reimplementing everything from scratch.

## 🚫 Commercial Use Disclaimer

The materials, code, and assets within this repository are intended solely for educational, training, or research purposes. They may not be utilized for commercial activities without explicit authorization. Any unauthorized commercial usage, distribution, or licensing of this repository's content is strictly forbidden. I am not the license holder for the original implementation. This is derived work. For detailed terms, please refer to the license section.


## News
- **[2025-05-27]**: Updated to LibTorch 2.7.0 for better compatibility and performance. Breaking changes in optimizer state management have been addressed.
- **[2025-05-26]**: The current goal of this repo is to move towards a permissive license. Some works has been done in this direction. However, as the major work package we must remove the rasterizer and replace it with the gsplat implementation.

## About this Project
This project is a derivative of the original Gaussian-Splatting software and is governed by the Gaussian-Splatting License, which can be found in the LICENSE file in this repository. The original software was developed by Inria and MPII.

Please be advised that the software in this repository cannot be used for commercial purposes without explicit consent from the original licensors, Inria and MPII.

## Build and Execution instructions
### Software Prerequisites
1. Linux (tested with Ubuntu 22.04), windows probably won't work.
2. CMake 3.24 or higher.
3. CUDA 11.7 or higher (might work with a lower version, has to be manually set and tested).
4. Python with development headers.
5. **LibTorch 2.7.0**: You can find the setup instructions in the libtorch section of this README.
6. Other dependencies will be handled by the CMake script.

### Hardware Prerequisites
1. NVIDIA GPU with CUDA support. Successfully tested so far are RTX 4090, RTX A5000, 3090Ti and A100. With 3080Ti there is an outstanding issue (#21) with larger datasets.
2. So far, the lowest compute capability tested was 8.0.

It might work with other NVIDIA GPUs as well, but these are mostly untested. If you do successfully run on such hardware please
post a message in the Discussions section of the repo.

### Build
```bash
git clone --recursive https://github.com/MrNeRF/gaussian-splatting-cuda
cd gaussian-splatting-cuda
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu118.zip  
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu118.zip -d external/
rm libtorch-cxx11-abi-shared-with-deps-2.7.0+cu118.zip
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j
```

## LibTorch Version Update

**Important**: This project has been updated to **LibTorch 2.7.0** (previously used 2.0.1). Key improvements include:

- **Enhanced Performance**: Better optimization and memory management
- **API Stability**: Updated to use the latest stable PyTorch C++ API
- **CUDA Compatibility**: Better integration with CUDA 11.8
- **Bug Fixes**: Resolved optimizer state management issues from earlier versions

### Migration Notes

If you're upgrading from a previous version:
1. **Download the new LibTorch version** using the updated build instructions above
2. **Clean your build directory**: `rm -rf build/` before rebuilding
3. **Compatibility**: The code has been updated to handle LibTorch 2.7.0 API changes automatically

### Dataset
The dataset is not included in this repository. You can download it from the original repository under the following link:
[tanks & trains](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).
Then unzip it in the data folder.

### Command-Line Options

The `3D Gaussian Splatting CUDA Implementation` provides a suite of command-line options to facilitate easy and customizable execution. Below are the available options:

### Core Options

- **-h, --help**  
  Display this help menu.

- **-d, --data_path [PATH]**  
  Specify the path to the training data.
 
- **-f, --force**  
    Force overwriting of output folder. If not set, the program will exit if the output folder already exists.
 
- **-o, --output_path [PATH]**  
  Specify the path to save the trained model. If this option is not specified, the trained model will be saved to the "output" folder located in the root directory of the project.

- **-i, --iter [NUM]**  
  Specify the number of iterations to train the model. Although the paper sets the maximum number of iterations at 30k, you'll likely need far fewer. Starting with 6k or 7k iterations should yield preliminary results. Outputs are saved every 7k iterations and also at the end of the training. Therefore, even if you set it to 5k iterations, an output will be generated upon completion.

### Example

To run the `3D Gaussian Splatting CUDA Implementation` with specified data path, output path, and iterations, use the following command:

```bash
$ ./build/gaussian_splatting_cuda -d /path/to/data -o /path/to/output -i 1000
```

### View the results
For now, you will need the SIBR view
```bash
git clone --recursive https://gitlab.inria.fr/sibr/sibr_core SIBR_viewers
cd SIBR_viewers
cmake -B build .
cmake --build build --target install --config Release -- -j 
cd ..
```
Then, you can view the results with:
```bash
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m output
```

## Contributions
Contributions are welcome! I want to make this a community project.

Some ideas for relative straight forward contributions:
- Revamp the README.
- Add a proper config file or cmd line config.

I want to get rid of some heavy dependencies:
- Replace glm with custom matrix operations
- Replace the few Eigen with some custom matrix operations

Advanced contributions or long term goals:
- Build a renderer to view training output in real time and to replace SIBR viewer.
- Look into [gtsfm](https://github.com/borglab/gtsfm) to replace colmap dependency
- CUDA optimization
- Build a proper viewer for the training output (maybe with XR support?).

Own ideas are welcome as well!

### Contribution Guidelines

Below are some guidelines to help ensure our project remains effective and consistent.

1. **Getting Started with Contributions**:
    - I've marked several beginner-friendly issues as **good first issues**. If you're new to the project, these are great places to start.
    - For those looking to contribute something not currently listed as an issue or propose something in the discussion section. You can direct message me on Twitter for a quick chat. Since there are not many contributors at the moment, I'm happy to discuss your ideas and help you get started.

2. **Before Submitting Your Pull Request**:
    - Ensure you've applied `clang-format` to maintain consistent coding style. There is in tools folder a git pre-commit hook. You can just copy it to .git/hooks/pre-commit. It will run clang-format before every commit.
    - We aim to minimize dependencies. If you're introducing a new one, it's essential to raise an issue for discussion first. There are ongoing efforts to reduce the number of dependencies, and your understanding in this area is appreciated.

3. **Key Principles for Contributions**:
    - **Speed**: We want to be the fastest 3D gaussian splatting implementation on this planet. Being lightning fast is key! I want instant training!
    - **Quality**: Always prioritize high-quality rendering output. Never compromise quality for speed.
    - **Usability**: We want to have a nice user experience. We're still perfecting this, and your contribution can make a difference!

4. **Dataset Contributions**:
    - If you have a unique dataset that you believe will be an excellent addition and that is eye popping, we'd love to see it! Remember, we're aiming to showcase exceptional datasets. We want to show off the best of the best. If you're unsure, feel free to raise an issue for discussion first.

Together, with your contributions, we can make this project stand out. Thank you for being a part of this journey!

## libtorch
Initially, I utilized libtorch to simplify the development process. Once the implementation is stable with libtorch, I will begin replacing torch elements with my custom CUDA implementation.
## MISC
Here is random collection of things that have to be described in README later on
- Needed for simple-knn: 
```bash sudo apt-get install python3-dev ```
 
## Citation and References
If you utilize this software or present results obtained using it, please reference the original work:

Kerbl, Bernhard; Kopanas, Georgios; Leimkühler, Thomas; Drettakis, George (2023). [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). ACM Transactions on Graphics, 42(4).

This will ensure the original authors receive the recognition they deserve.

## License

This project is licensed under the Gaussian-Splatting License - see the [LICENSE](LICENSE) file for details.

Follow me on Twitter if you want to know more about the latest development: https://twitter.com/janusch_patas
