# LichtFeld Studio

<div align="center">

**A high-performance C++ and CUDA implementation of 3D Gaussian Splatting**

[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?logo=discord&logoColor=white)](https://discord.gg/TbxJST2BbC)
[![Website](https://img.shields.io/badge/Website-Lichtfeld%20Studio-blue)](https://mrnerf.github.io/lichtfeld-studio-web/)
[![Papers](https://img.shields.io/badge/Papers-Awesome%203DGS-orange)](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)
[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.8+-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![C++](https://img.shields.io/badge/C++-23-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/23)

<img src="docs/viewer_demo.gif" alt="3D Gaussian Splatting Viewer" width="85%"/>

[**Quick Start**](#quick-start) •
[**Installation**](#installation) •
[**Usage**](#usage) •
[**Results**](#benchmark-results) •
[**Community**](#community--support)

</div>

---

## Overview

LichtFeld Studio is a high-performance implementation of 3D Gaussian Splatting that leverages modern C++23 and CUDA 12.8+ for optimal performance. Built with a modular architecture, it provides both training and real-time visualization capabilities for neural rendering research and applications.

### Key Features

- **2.4x faster rasterization** (winner of first bounty by Florian Hahlbohm)
- **MCMC optimization strategy** for improved convergence
- **Real-time interactive viewer** with OpenGL rendering
- **Modular architecture** with separate core, training, and rendering components
- **Multiple rendering modes** including RGB, depth, and combined views
- **Bilateral grid appearance modeling** for handling per-image variations

## Community & Support

Join our growing community for discussions, support, and updates:

- **[Discord Community](https://discord.gg/TbxJST2BbC)** - Get help, share results, and discuss development
- **[Website](https://mrnerf.com)** - Visit our website for more resources
- **[Awesome 3D Gaussian Splatting](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)** - Comprehensive paper list
- **[@janusch_patas](https://twitter.com/janusch_patas)** - Follow for the latest updates

## Active Bounty

### Second Bounty: Better 3DGS Initialization and Training without Densification
**$2,600 + $500 Bonus Challenge**

Details: [Issue #284](https://github.com/MrNeRF/gaussian-splatting-cuda/issues/284)

Previous winner: [Florian Hahlbohm](https://github.com/MrNeRF/gaussian-splatting-cuda/pull/245) (2.4x rasterizer speedup)

## Quick Start

```bash
# Clone and build (Linux)
git clone https://github.com/MrNeRF/gaussian-splatting-cuda
cd gaussian-splatting-cuda

# Download LibTorch
wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip  
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip -d external/

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -- -j$(nproc)

# Train on sample data
./build/gaussian_splatting_cuda -d data/garden -o output/garden --eval
```

## Installation

### Requirements

#### Software
- **OS**: Linux (Ubuntu 22.04+) or Windows
- **CMake**: 3.24 or higher
- **Compiler**: C++23 compatible (GCC 14+ or Clang 17+)
- **CUDA**: 12.8 or higher (required)
  
- **LibTorch**: 2.7.0 (setup instructions below)
- **vcpkg**: For dependency management

#### Hardware
- **GPU**: NVIDIA GPU with compute capability 7.5+
- **VRAM**: Minimum 8GB recommended
- **Tested GPUs**: RTX 4090, RTX A5000, RTX 3090Ti, A100, RTX 2060 SUPER

### Build Instructions

<details>
<summary><b>Linux Build</b></summary>

```bash
# Set up vcpkg (one-time setup)
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && ./bootstrap-vcpkg.sh -disableMetrics && cd ..

## If you want you can specify vcpkg locally without globally setting env variable (see -DCMAKE_TOOLCHAIN_FILE version)
export VCPKG_ROOT=/path/to/vcpkg  # Add to ~/.bashrc

# Clone repository
git clone https://github.com/MrNeRF/gaussian-splatting-cuda
cd gaussian-splatting-cuda

# Download LibTorch 2.7.0 with CUDA 12.8
wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip  
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip -d external/
rm libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja

## Or if you want you can specify your own vcpkg 
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake" -G Ninja 

cmake --build build -- -j$(nproc)
```

</details>

<details>
<summary><b>Windows Build</b></summary>

Run in <u>**x64 native tools command prompt for VS**</u>:

```bash
# Set up vcpkg (one-time setup)
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && .\bootstrap-vcpkg.bat -disableMetrics && cd ..

## If you want you can specify vcpkg locally without globally setting env variable (see -DCMAKE_TOOLCHAIN_FILE version)
set VCPKG_ROOT=%CD%\vcpkg

# Clone repository
git clone https://github.com/MrNeRF/gaussian-splatting-cuda
cd gaussian-splatting-cuda

# Create directories
if not exist external mkdir external
if not exist external\debug mkdir external\debug
if not exist external\release mkdir external\release

# Download LibTorch (Debug)
curl -L -o libtorch-debug.zip https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.7.0%2Bcu128.zip
tar -xf libtorch-debug.zip -C external\debug
del libtorch-debug.zip

# Download LibTorch (Release)
curl -L -o libtorch-release.zip https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.7.0%2Bcu128.zip
tar -xf libtorch-release.zip -C external\release
del libtorch-release.zip

# Build 

## Or if you want you can specify your own vcpkg 
# cmake -B build -DCMAKE_BUILD_TYPE=Release -G ninja -DCMAKE_TOOLCHAIN_FILE="<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake"

# Ninja should be included with Visual Studio installation, 
# if not you can either install it, 
# or ignore this flag and use native generator - Building time might be extended
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja

cmake --build build -j
```

</details>

<details>
<summary><b>Docker Build</b></summary>

```bash
# Build and start container
./docker/run_docker.sh -bu 12.8.0

# Build without cache
./docker/run_docker.sh -n

# Stop containers
./docker/run_docker.sh -c
```

</details>

### Compiler Setup

<details>
<summary><b>Ubuntu 24.04+ (GCC 14)</b></summary>

```bash
# Install GCC 14
sudo apt update
sudo apt install gcc-14 g++-14 gfortran-14

# Set as default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 60
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 60
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

</details>

<details>
<summary><b>Ubuntu 22.04 (Build GCC 14 from source)</b></summary>

```bash
# Install dependencies
sudo apt install build-essential libmpfr-dev libgmp3-dev libmpc-dev -y

# Download and build GCC
wget http://ftp.gnu.org/gnu/gcc/gcc-14.1.0/gcc-14.1.0.tar.gz
tar -xf gcc-14.1.0.tar.gz
cd gcc-14.1.0

# Configure and build (1-2 hours)
./configure --prefix=/usr/local/gcc-14.1.0 --enable-languages=c,c++ --disable-multilib
make -j$(nproc)
sudo make install

# Set up alternatives
sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/gcc-14.1.0/bin/gcc 14
sudo update-alternatives --install /usr/bin/g++ g++ /usr/local/gcc-14.1.0/bin/g++ 14
```

</details>

## Usage

### Dataset Preparation

Download and extract the Tanks & Trains dataset:

```bash
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip -d data/
```

### Training

Basic training:
```bash
./build/gaussian_splatting_cuda -d data/garden -o output/garden
```

Training with evaluation and visualization:
```bash
./build/gaussian_splatting_cuda \
    -d data/garden \
    -o output/garden \
    --eval \
    --save-eval-images \
    --render-mode RGB_D \
    -i 30000
```

MCMC strategy with limited Gaussians:
```bash
./build/gaussian_splatting_cuda \
    -d data/garden \
    -o output/garden \
    --strategy mcmc \
    --max-cap 500000
```

### Command-Line Options

#### Required
- `-d, --data-path [PATH]` - Path to training data with COLMAP reconstruction

#### Training Configuration
- `-o, --output-path [PATH]` - Output directory (default: `./output`)
- `-i, --iter [NUM]` - Training iterations (default: 30000)
- `-r, --resize_factor [NUM]` - Image resolution factor (default: 1)
- `--strategy [mcmc|default]` - Optimization strategy (default: `mcmc`)
- `--max-cap [NUM]` - Maximum Gaussians for MCMC (default: 1000000)

#### Evaluation
- `--eval` - Enable evaluation during training
- `--save-eval-images` - Save evaluation images
- `--test-every [NUM]` - Test/validation split ratio (default: 8)

#### Visualization
- `--headless` - Run without GUI (terminal-only mode)

#### Advanced Options
- `--bilateral-grid` - Enable appearance modeling
- `--steps-scaler [NUM]` - Scale training steps for multiple checkpoints
- See `--help` for complete list of options

### LPIPS Model Details

The implementation uses `weights/lpips_vgg.pt`, exported from `torchmetrics` with:
- **Network**: VGG with ImageNet pretrained weights
- **Input range**: [-1, 1] (conversion handled internally)
- **Normalization**: Included in model

## Project Architecture

```
gaussian-splatting-cuda/
├── src/
│   ├── core/          # Foundation (data structures, utilities)
│   ├── geometry/      # Geometric operations
│   ├── loader/        # Dataset loading (COLMAP, PLY, Blender)
│   ├── training/      # Training pipeline and strategies
│   ├── rendering/     # CUDA/OpenGL rendering
│   └── visualizer/    # Interactive GUI
├── gsplat/            # Optimized rasterization backend
├── fastgs/            # Fast Gaussian splatting kernels
└── parameter/         # JSON configuration files
```

## Contributing

We welcome contributions! See our [Contributing Guidelines](CONTRIBUTING.md).

### Getting Started
- Check issues labeled **good first issue**
- Join our [Discord](https://discord.gg/TbxJST2BbC) for discussions
- Use the pre-commit hook: `cp tools/pre-commit .git/hooks/`

### Development Requirements
- C++23 compatible compiler (GCC 14+ or Clang 17+)
- CUDA 12.8+ for GPU development
- Apply `clang-format` for code style

## Acknowledgments

This implementation builds upon:
- **[gsplat](https://github.com/nerfstudio-project/gsplat)** - Optimized CUDA rasterization backend
- **[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)** - Original work by Kerbl et al.

## Citation

```bibtex
@software{lichtfeld2025,
  author    = {LichtFeld Studio},
  title     = {A high-performance C++ and CUDA implementation of 3D Gaussian Splatting},
  year      = {2025},
  url       = {https://github.com/MrNeRF/gaussian-splatting-cuda}
}
```

## License

This project is licensed under GPLv3. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Connect with us:** [Website](https://mrnerf.com) • [Discord](https://discord.gg/TbxJST2BbC) • [Twitter](https://twitter.com/janusch_patas)

</div>
