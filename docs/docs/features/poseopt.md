# Pose Optimization
Slightly misaligned poses can lead to poor details in reconstructions. To address this, we provide a pose optimization feature that refines camera poses during training. This feature is based on the [3R-GS paper](https://arxiv.org/abs/2504.04294).

## How to Use
LFS provides two modes of pose optimization:

- Direct: This mode optimizes the camera pose offsets directly. Can be enabled with the `--pose-opt direct` flag.
- MLP: This mode optimizes camera embeddings which are passed through a small neural network (which is also trained) that predicts pose offsets. Can be enabled with the `--pose-opt mlp` flag. This method should provide better optimization results, but is slower and uses more memory.
