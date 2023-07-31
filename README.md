# Gaussian splatting implementation with CUDA
This is my attempt to learn more about the wonderful paper 3D gaussian splatting by reimplementing everthing from scratch. 
This repo is highly work in progress.

### libtorch
In the beginning I use libtorch to make my life easier. When everything works with libtorch, I will start replacing 
torch elements with my own custom cuda implementation.


Download the libtorch library using the following command:

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
```

This will download a zip file named `libtorch-shared-with-deps-latest.zip`. To extract this zip file, use the command:

```bash
unzip libtorch-shared-with-deps-latest.zip -d external/
rm libtorch-shared-with-deps-latest.zip
```

This will create a folder named `libtorch` in the `external` directory of your project.


###  Original Repository
https://github.com/graphdeco-inria/gaussian-splatting