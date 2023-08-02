// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#include "rasterizer.cuh"

torch::Tensor rasterize_gaussians(torch::Tensor means3D,
                                  torch::Tensor means2D,
                                  torch::Tensor sh,
                                  torch::Tensor colors_precomp,
                                  torch::Tensor opacities,
                                  torch::Tensor scales,
                                  torch::Tensor rotations,
                                  torch::Tensor cov3Ds_precomp,
                                  GaussianRasterizationSettings raster_settings) {
    return _RasterizeGaussians::apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings);
}
