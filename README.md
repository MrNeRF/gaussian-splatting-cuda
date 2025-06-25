# 3D Gaussian Splatting for Real-Time Radiance Field Rendering - C++ and CUDA Implementation

[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?logo=discord&logoColor=white)](https://discord.gg/TbxJST2BbC)
[![Website](https://img.shields.io/badge/Website-mrnerf.com-blue)](https://mrnerf.com)
[![Papers](https://img.shields.io/badge/Papers-Awesome%203DGS-orange)](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)

A high-performance C++ and CUDA implementation of 3D Gaussian Splatting, built upon the [gsplat](https://github.com/nerfstudio-project/gsplat) rasterization backend.

<img src="docs/viewer_demo.gif" alt="3D Gaussian Splatting Viewer" width="80%"/>

## News
- **[2025-06-20]**: Added interactive viewer with real-time visualization during training by @panxkun.
- **[2025-06-19]**: Metrics are now on par with gsplat-mcmc. Gsplat evals on downscaled png images whereas I used jpgs.
- **[2025-06-15]**: Different render modes exposed, refactors, added bilateral grid.
- **[2025-06-13]**: Metrics are getting very close to gsplat-mcmc. LPIPS and time estimates are not comparable as of now.
- **[2025-06-10]**: Fixed some issues. We are closing the gap to the gsplat metrics. However, there is still a small mismatch.
- **[2025-06-04]**: Added MCMC strategy with `--max-cap` command line option for controlling maximum Gaussian count.
- **[2025-06-03]**: Switched to Gsplat backend and updated license to Apache 2.0.
- **[2024-05-27]**: Updated to LibTorch 2.7.0 for better compatibility and performance. Breaking changes in optimizer state management have been addressed.

### LPIPS Model
The implementation uses `weights/lpips_vgg.pt`, which is exported from `torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity` with:
- **Network type**: VGG
- **Normalize**: False (model expects inputs in [-1, 1] range)
- **Model includes**: VGG backbone with pretrained ImageNet weights and the scaling normalization layer

**Note**: While the model was exported with `normalize=False`, the C++ implementation handles the [0,1] to [-1,1] conversion internally during LPIPS computation, ensuring compatibility with images loaded in [0,1] range.

| Scene    | Iteration | PSNR          | SSIM         | LPIPS        | Num Gaussians |
| -------- | --------- | ------------- | ------------ | ------------ |---------------|
| garden   | 30000     | 27.538504     | 0.866146     | 0.148426     | 1000000       |
| bicycle  | 30000     | 25.771051     | 0.790709     | 0.244115     | 1000000       |
| stump    | 30000     | 27.141726     | 0.805854     | 0.246617     | 1000000       |
| bonsai   | 30000     | 32.586533     | 0.953505     | 0.224543     | 1000000       |
| counter  | 30000     | 29.346529     | 0.923511     | 0.223990     | 1000000       |
| kitchen  | 30000     | 31.840155     | 0.938906     | 0.141826     | 1000000       |
| room     | 30000     | 32.511021     | 0.938708     | 0.253696     | 1000000       |
| **mean** | **30000** | **29.533646** | **0.888191** | **0.211888** | **1000000**   |

For reference, here are the metrics for the official gsplat-mcmc implementation below. However, the
lpips results are not directly comparable, as the gsplat-mcmc implementation uses a different lpips model.

| Scene    | Iteration | PSNR          | SSIM         | LPIPS        | Num Gaussians |
| -------- | --------- | ------------- | ------------ | ------------ | ------------- |
| garden   | 30000     | 27.307266     | 0.854643     | 0.103883     | 1000000       |
| bicycle  | 30000     | 25.615253     | 0.774689     | 0.182401     | 1000000       |
| stump    | 30000     | 26.964493     | 0.789816     | 0.162758     | 1000000       |
| bonsai   | 30000     | 32.735737     | 0.953360     | 0.105922     | 1000000       |
| counter  | 30000     | 29.495266     | 0.924103     | 0.129898     | 1000000       |
| kitchen  | 30000     | 31.660593     | 0.935315     | 0.087113     | 1000000       |
| room     | 30000     | 32.265732     | 0.937518     | 0.132472     | 1000000       |
| **mean** | **30000** | **29.434906** | **0.881349** | **0.129207** | **1000000**   |

## Community & Support

Join our growing community for discussions, support, and updates:
- 💬 **[Discord Community](https://discord.gg/TbxJST2BbC)** - Get help, share results, and discuss development
- 🌐 **[mrnerf.com](https://mrnerf.com)** - Visit our website for more resources
- 📚 **[Awesome 3D Gaussian Splatting](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)** - Comprehensive paper list and resources
- 🐦 **[@janusch_patas](https://twitter.com/janusch_patas)** - Follow for the latest updates

## Build and Execution Instructions

### Software Prerequisites
1. **Operating System**:
    - **Linux** (tested with Ubuntu 22.04)
    - **Windows 10/11** (tested with Visual Studio 2022)
2. **CMake** 3.24 or higher (add to PATH)
3. **CUDA Toolkit**:
    - Version 11.8 or higher. For Windows, ensure it integrates with Visual Studio 2022.
4. **Python**:
    - Version 3.x with development headers (ensure Python is in PATH and `python-config` or `python3-config` is available for CMake to find it on Linux). On Windows, CMake should find a valid Python installation.
5. **LibTorch 2.7.0 (C++ API of PyTorch)**:
    - Setup instructions specific to your OS are below. Ensure you download the version compatible with your CUDA Toolkit version (e.g., cu118 for CUDA 11.8).
6. **C++ Compiler**:
    - **Linux**: GCC or Clang with C++17 support.
    - **Windows**: Visual Studio 2022 (Desktop development with C++ workload).
7. **Git** (for cloning and submodule management).
8. Other dependencies (GLFW, GLM, TBB, etc.) are handled automatically by CMake.
    - **Submodules**: External libraries like `json`, `args`, `glfw`, `glm`, `glad`, `imgui` are included as git submodules. Ensure these are initialized correctly (see build instructions).
        - *Note on `json` (nlohmann/json submodule)*: If you encounter CMake policy errors related to this submodule during configuration (especially with older CMake versions or projects), you might need to set `CMAKE_POLICY_VERSION_MINIMUM` to `3.5` or a similar compatible version. In CMake GUI, you can do this by clicking "Add Entry", setting Name: `CMAKE_POLICY_VERSION_MINIMUM`, Type: `STRING`, Value: `3.5`. For command-line CMake, you can add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.
    - **TBB (Threading Building Blocks)**:
        - On Linux, CMake will try to find TBB via `find_package`. Ensure it's installed (e.g., `sudo apt-get install libtbb-dev`).
        - On Windows, TBB might need to be installed separately, for example, via the Intel oneAPI Base Toolkit. If CMake cannot find TBB, you may need to set the `TBB_DIR` CMake variable to point to your TBB installation's CMake configuration directory (e.g., `C:/Program Files (x86)/Intel/oneAPI/tbb/latest/lib/cmake/TBB`).

### Hardware Prerequisites
1. **NVIDIA GPU** with CUDA support
    - Successfully tested: RTX 4090, RTX A5000, RTX 3090Ti, A100
    - Known issue with RTX 3080Ti on larger datasets (see #21)
2. Minimum recommended compute capability: 7.0 (Volta), 8.0+ (Ampere or newer) preferred for full performance.

> If you successfully run on other hardware, please share your experience in the Discussions section!

### Build Instructions

**Important First Step (All Platforms):**

This project uses git submodules for some external dependencies. After cloning, or if you pulled changes that might affect submodules, ensure they are initialized and updated:
```bash
# If you just cloned:
git clone https://github.com/MrNeRF/gaussian-splatting-cuda
cd gaussian-splatting-cuda
git submodule update --init --recursive

# If you already cloned and just need to update submodules:
# cd gaussian-splatting-cuda
# git submodule update --recursive
```

#### For Linux

```bash
# Assumes you are in the gaussian-splatting-cuda directory
# and submodules are initialized (see "Important First Step" above)

# Download and setup LibTorch (for CUDA 11.8)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu118.zip -d external/
# It's recommended to move the unzipped 'libtorch' folder to 'external/libtorch' directly
# e.g., mv external/libtorch-shared-with-deps external/libtorch
rm libtorch-cxx11-abi-shared-with-deps-2.7.0+cu118.zip

# Build the project
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) # or cmake --build . --config Release -- -j$(nproc)
```

#### For Windows (Visual Studio 2022)

1.  **Clone the repository and initialize submodules:**
    Follow the "Important First Step (All Platforms)" instructions above. Make sure you are in the `gaussian-splatting-cuda` directory.

2.  **Download and setup LibTorch (for CUDA 11.8, Release version):**
    *   Go to the [PyTorch website](https://pytorch.org/get-started/locally/) and select the appropriate options for your setup:
        *   PyTorch Build: Stable
        *   Your OS: Windows
        *   Package: LibTorch
        *   Language: C++/Java
        *   Compute Platform: CUDA 11.8 (or your installed CUDA version, e.g., CUDA 12.1)
    *   Download the "Release" version zip file. For example, for CUDA 11.8, it might be `libtorch-win-shared-with-deps-2.7.0%2Bcu118.zip`.
    *   Extract the zip file into the `external/` directory in the project root.
    *   Rename the extracted folder (e.g., `libtorch-shared-with-deps`) to `libtorch`. So, the path should be `external/libtorch`.

3.  **Configure with CMake:**
    *   Open CMake GUI or use the command line.
    *   Set the source code directory to the project root.
    *   Set the build directory (e.g., `build` inside the project root).
    *   Click "Configure".
    *   Specify the generator: "Visual Studio 17 2022".
    *   Ensure CMake correctly finds CUDA, Python, and Torch.
        *   If Torch is not found, you might need to set `Torch_DIR` manually in CMake to `external/libtorch/share/cmake/Torch`.
    *   Set `CMAKE_BUILD_TYPE` to `Release`.
    *   Click "Generate".

4.  **Build with Visual Studio:**
    *   Open the generated `.sln` file in the build directory with Visual Studio 2022.
    *   Select the "Release" configuration.
    *   Build the `gaussian_splatting_cuda` target (or "Build All").
    *   The executable will be in the `build/Release` directory.

    Alternatively, from the command line after CMake generation:
    ```bash
    cd build
    cmake --build . --config Release -- -j%NUMBER_OF_PROCESSORS%
    ```
    (Replace `%NUMBER_OF_PROCESSORS%` with the number of cores you want to use, e.g., `cmake --build . --config Release -- -j8`)


## LibTorch 2.7.0

This project uses **LibTorch 2.7.0** for optimal performance and compatibility. Please ensure you download the correct version for your operating system and CUDA setup as described in the build instructions.

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

### Required Options

- **`-d, --data-path [PATH]`**  
  Path to the training data containing COLMAP sparse reconstruction (required)

### Output Options

- **`-o, --output-path [PATH]`**  
  Path to save the trained model (default: `./output`)

### Training Options

- **`-i, --iter [NUM]`**  
  Number of training iterations (default: 30000)
    - Paper suggests 30k, but 6k-7k often yields good preliminary results
    - Outputs are saved every 7k iterations and at completion

- **`-r, --resolution [NUM]`**  
  Set the resolution for training images
    - -1: Use original resolution (default)
    - Positive values: Target resolution for image loading

- **`--steps-scaler [NUM]`**  
  Scale all training steps by this factor (default: 1)
  - Multiplies iterations, refinement steps, and evaluation/save intervals
  - Creates multiple scaled checkpoints for each original step

### MCMC-Specific Options

- **`--max-cap [NUM]`**  
  Maximum number of Gaussians for MCMC strategy (default: 1000000)
    - Controls the upper limit of Gaussian splats during training
    - Useful for memory-constrained environments

### Dataset Configuration

- **`--images [FOLDER]`**  
  Images folder name (default: `images`)
    - Options: `images`, `images_2`, `images_4`, `images_8`
    - Mip-NeRF 360 dataset uses different resolutions

- **`--test-every [NUM]`**  
  Every N-th image is used as a test image (default: 8)
    - Used for train/validation split

### Evaluation Options

- **`--eval`**  
  Enable evaluation during training
    - Computes metrics (PSNR, SSIM, LPIPS) at specified steps
    - Evaluation steps defined in `parameter/optimization_params.json`

- **`--save-eval-images`**  
  Save evaluation images during training
    - Requires `--eval` to be enabled
    - Saves comparison images and depth maps (if applicable)

### Render Mode Options

- **`--render-mode [MODE]`**  
  Render mode for training and evaluation (default: `RGB`)
    - `RGB`: Color only
    - `D`: Accumulated depth only
    - `ED`: Expected depth only
    - `RGB_D`: Color + accumulated depth
    - `RGB_ED`: Color + expected depth

### Visualization Options

- **`-v, --viz`**  
  Enable the Visualization mode
    - Displays the current state of the Gaussian splatting in a window
    - Useful for debugging and monitoring training progress
    
### Advanced Options

- **`--bilateral-grid`**  
  Enable bilateral grid for appearance modeling
    - Helps with per-image appearance variations
    - Adds TV (Total Variation) regularization

- **`--sh-degree-interval [NUM]`**  
  Interval for increasing spherical harmonics degree
  - Controls how often SH degree is incremented during training

- **`-h, --help`**  
  Display the help menu

### Example Usage

Basic training:
```bash
./build/gaussian_splatting_cuda -d /path/to/data -o /path/to/output
```

MCMC training with limited Gaussians:
```bash
./build/gaussian_splatting_cuda -d /path/to/data -o /path/to/output --max-cap 500000
```

Training with evaluation and custom settings:
```bash
./build/gaussian_splatting_cuda \
    -d data/garden \
    -o output/garden \
    --images images_4 \
    --test-every 8 \
    --eval \
    --save-eval-images \
    --render-mode RGB_D \
    -i 30000
```

Force overwrite existing output:
```bash
./build/gaussian_splatting_cuda -d data/garden -o output/garden -f
```

Training with step scaling for multiple checkpoints:
```bash
./build/gaussian_splatting_cuda \
    -d data/garden \
    -o output/garden \
    --steps-scaler 3 \
    -i 10000
```

## Configuration Files

The implementation uses JSON configuration files located in the `parameter/` directory:

### `optimization_params.json`
Controls training hyperparameters including:
- Learning rates for different components
- Regularization weights
- Refinement schedules
- Evaluation and save steps
- Render mode settings
- Bilateral grid parameters

Key parameters can be overridden via command-line options.

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