# 3D Gaussian Splatting for Real-Time Radiance Field Rendering - C++ and CUDA Implementation

[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?logo=discord&logoColor=white)](https://discord.gg/TbxJST2BbC)
[![Website](https://img.shields.io/badge/Website-mrnerf.com-blue)](https://mrnerf.com)
[![Papers](https://img.shields.io/badge/Papers-Awesome%203DGS-orange)](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)

A high-performance C++ and CUDA implementation of 3D Gaussian Splatting, built upon the [gsplat](https://github.com/nerfstudio-project/gsplat) rasterization backend.

## News
- **[2025-06-10]**: Fixed some issues. We are closing the gap to the gsplat metrics. However, there is still a small mismatch.
- **[2025-06-04]**: Added MCMC strategy with `--max-cap` command line option for controlling maximum Gaussian count.
- **[2025-06-03]**: Switched to Gsplat backend and updated license.
- **[2024-05-27]**: Updated to LibTorch 2.7.0 for better compatibility and performance. Breaking changes in optimizer state management have been addressed.
- **[2024-05-26]**: The current goal of this repo is to move towards a permissive license. Major work has been done to replace the rasterizer with the gsplat implementation.

## Metrics
Currently the implementation does not achieve on par results with gsplat-mcmc, but it is a work in progress.
It is just a matter of time to fix the bug. Help is welcome :) The metrics for the mcmc strategy are as follows:

### LPIPS Model
The implementation uses `weights/lpips_vgg.pt`, which is exported from `torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity` with:
- **Network type**: VGG
- **Normalize**: False (model expects inputs in [-1, 1] range)
- **Model includes**: VGG backbone with pretrained ImageNet weights and the scaling normalization layer

**Note**: While the model was exported with `normalize=False`, the C++ implementation handles the [0,1] to [-1,1] conversion internally during LPIPS computation, ensuring compatibility with images loaded in [0,1] range.

| Scene    | Iteration | PSNR          | SSIM         | LPIPS        | Time per Image | Num Gaussians |
| -------- | --------- | ------------- | ------------ | ------------ | -------------- | ------------- |
| garden   | 30000     | 27.112114     | 0.854833     | 0.157624     | 0.304765       | 1000000       |
| bicycle  | 30000     | 25.047745     | 0.767729     | 0.254825     | 0.293618       | 1000000       |
| stump    | 30000     | 26.554749     | 0.784203     | 0.263013     | 0.296536       | 1000000       |
| bonsai   | 30000     | 32.534199     | 0.948675     | 0.246924     | 0.436188       | 1000000       |
| counter  | 30000     | 29.187017     | 0.915823     | 0.242159     | 0.441259       | 1000000       |
| kitchen  | 30000     | 31.680832     | 0.933897     | 0.154965     | 0.449078       | 1000000       |
| room     | 30000     | 32.211632     | 0.930754     | 0.273719     | 0.413519       | 1000000       |
| **mean** | **30000** | **29.189755** | **0.876559** | **0.227604** | **0.376423**   | **1000000**   |

## Community & Support

Join our growing community for discussions, support, and updates:
- 💬 **[Discord Community](https://discord.gg/TbxJST2BbC)** - Get help, share results, and discuss development
- 🌐 **[mrnerf.com](https://mrnerf.com)** - Visit our website for more resources
- 📚 **[Awesome 3D Gaussian Splatting](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)** - Comprehensive paper list and resources
- 🐦 **[@janusch_patas](https://twitter.com/janusch_patas)** - Follow for the latest updates

## Build and Execution Instructions

### Software Prerequisites
1. **Linux** (tested with Ubuntu 22.04) - Windows is currently not supported
2. **CMake** 3.24 or higher
3. **CUDA** 11.8 or higher (may work with lower versions with manual configuration)
4. **Python** with development headers
5. **LibTorch 2.7.0** - Setup instructions below
6. Other dependencies are handled automatically by CMake

### Hardware Prerequisites
1. **NVIDIA GPU** with CUDA support
    - Successfully tested: RTX 4090, RTX A5000, RTX 3090Ti, A100
    - Known issue with RTX 3080Ti on larger datasets (see #21)
2. Minimum compute capability: 8.0

> If you successfully run on other hardware, please share your experience in the Discussions section!

### Build Instructions

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/MrNeRF/gaussian-splatting-cuda
cd gaussian-splatting-cuda

# Download and setup LibTorch
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu118.zip  
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu118.zip -d external/
rm libtorch-cxx11-abi-shared-with-deps-2.7.0+cu118.zip

# Build the project
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j
```

## LibTorch 2.7.0

This project uses **LibTorch 2.7.0** for optimal performance and compatibility:

- **Enhanced Performance**: Improved optimization and memory management
- **API Stability**: Latest stable PyTorch C++ API
- **CUDA Compatibility**: Better integration with CUDA 11.8+
- **Bug Fixes**: Resolved optimizer state management issues

### Upgrading from Previous Versions
1. Download the new LibTorch version using the build instructions
2. Clean your build directory: `rm -rf build/`
3. Rebuild the project

## Dataset

Download the dataset from the original repository:
[Tanks & Trains Dataset](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)

Extract it to the `data` folder in the project root.

## Command-Line Options

### Core Options

- **`-h, --help`**  
  Display the help menu

- **`-d, --data-path [PATH]`**  
  Path to the training data (required)

- **`-o, --output-path [PATH]`**  
  Path to save the trained model (default: `./output`)

- **`-i, --iter [NUM]`**  
  Number of training iterations (default: 30000)
    - Paper suggests 30k, but 6k-7k often yields good preliminary results
    - Outputs are saved every 7k iterations and at completion

- **`-f, --force`**  
  Force overwrite of existing output folder

- **`-r, --resolution [NUM]`**  
  Set the resolution for training images

### MCMC-Specific Options

- **`--max-cap [NUM]`**  
  Maximum number of Gaussians for MCMC strategy (default: 1000000)
    - Controls the upper limit of Gaussian splats during training
    - Useful for memory-constrained environments

### Example Usage

Basic training:
```bash
./build/gaussian_splatting_cuda -d /path/to/data -o /path/to/output -i 10000
```

MCMC training with limited Gaussians:
```bash
./build/gaussian_splatting_cuda -d /path/to/data -o /path/to/output -i 10000 --max-cap 500000
```

## Contribution Guidelines

We welcome contributions! Here's how to get started:

1. **Getting Started**:
    - Check out issues labeled as **good first issues** for beginner-friendly tasks
    - For new ideas, open a discussion or join our [Discord](https://discord.gg/TbxJST2BbC)

2. **Before Submitting a PR**:
    - Apply `clang-format` for consistent code style
    - Use the pre-commit hook: `cp tools/pre-commit .git/hooks/`
    - Discuss new dependencies in an issue first - we aim to minimize dependencies

## Acknowledgments

This implementation builds upon several key projects:

- **[gsplat](https://github.com/nerfstudio-project/gsplat)**: We use gsplat's highly optimized CUDA rasterization backend, which provides significant performance improvements and better memory efficiency.

- **Original 3D Gaussian Splatting**: Based on the groundbreaking work by Kerbl et al.

## Citation

If you use this software in your research, please cite the original work:

```bibtex
@article{kerbl3Dgaussians,
  author    = {Kerbl, Bernhard and Kopanas, Georgios and Leimkühler, Thomas and Drettakis, George},
  title     = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal   = {ACM Transactions on Graphics},
  number    = {4},
  volume    = {42},
  month     = {July},
  year      = {2023},
  url       = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## License

See LICENSE file for details.

---

**Connect with us:**
- 🌐 Website: [mrnerf.com](https://mrnerf.com)
- 📚 Papers: [Awesome 3D Gaussian Splatting](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)
- 💬 Discord: [Join our community](https://discord.gg/TbxJST2BbC)
- 🐦 Twitter: Follow [@janusch_patas](https://twitter.com/janusch_patas) for development updates
