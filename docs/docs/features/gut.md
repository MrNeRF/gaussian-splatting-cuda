# 3DGUT

3DGUT (3D Gaussian Unscented Transform) is an alternative method of rendering proposed by NVIDIA Reasearch that uses raytracing instead of rasterization. Most significantly, it allows rendering and training with nonlinear projections, like camera models with distortion.

## When to use 3DGUT
Use 3DGUT when your COLMAP camera model is not PINHOLE or SIMPLE_PINHOLE.

:::warning
RealityScan (formerly RealityCapture) may export datasets with a camera model that has distortion with images that are undistorted. If you attempt to use 3DGUT with a dataset like this, the results will be incorrect. The `--rc` flag can be used to ignore the distortion parameters in this case.
:::

## Supported Camera Models
- SIMPLE_PINHOLE
- PINHOLE
- SIMPLE_RADIAL
- RADIAL
- OPENCV
- FULL_OPENCV
- OPENCV_FISHEYE
- RADIAL_FISHEYE
- SIMPLE_RADIAL_FISHEYE

## References
- [3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting](https://research.nvidia.com/labs/toronto-ai/3DGUT/) - Original paper and project page
- [3dgrut Repository](https://github.com/nv-tlabs/3dgrut) - Reference implementation
- [gsplat 3DGUT pull request](https://github.com/nerfstudio-project/gsplat/pull/667) - Implementation in gsplat which LFS's 3DGUT support is based on