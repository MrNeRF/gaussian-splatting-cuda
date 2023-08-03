// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include <torch/torch.h>

static const double C0 = 0.28209479177387814;
static const double C1 = 0.4886025119029199;
static std::vector<double> C2 = {1.0925484305920792,
                                 -1.0925484305920792,
                                 0.31539156525252005,
                                 -1.0925484305920792,
                                 0.5462742152960396};
static std::vector<double> C3 = {-0.5900435899266435,
                                 2.890611442640554,
                                 -0.4570457994644658,
                                 0.3731763325901154,
                                 -0.4570457994644658,
                                 1.445305721320277,
                                 -0.5900435899266435};
static std::vector<double> C4 = {2.5033429417967046,
                                 -1.7701307697799304,
                                 0.9461746957575601,
                                 -0.6690465435572892,
                                 0.10578554691520431,
                                 -0.6690465435572892,
                                 0.47308734787878004,
                                 -1.7701307697799304,
                                 0.6258357354491761};

inline torch::Tensor Eval_sh(int deg, const torch::Tensor& sh, const torch::Tensor& dirs) {
    assert(deg <= 4 && deg >= 0);
    int coeff = (deg + 1) * (deg + 1);
    assert(sh.size(-1) >= coeff);

    torch::Tensor result = C0 * sh.index({torch::indexing::Ellipsis, 0});
    if (deg > 0) {
        auto x = dirs.index({torch::indexing::Ellipsis, 0});
        auto y = dirs.index({torch::indexing::Ellipsis, 1});
        auto z = dirs.index({torch::indexing::Ellipsis, 2});
        result = result - C1 * y * sh.index({torch::indexing::Ellipsis, 1}) +
                 C1 * z * sh.index({torch::indexing::Ellipsis, 2}) -
                 C1 * x * sh.index({torch::indexing::Ellipsis, 3});
        if (deg > 1) {
            auto xx = x * x, yy = y * y, zz = z * z;
            auto xy = x * y, yz = y * z, xz = x * z;
            result = result + C2[0] * xy * sh.index({torch::indexing::Ellipsis, 4}) +
                     C2[1] * yz * sh.index({torch::indexing::Ellipsis, 5}) +
                     C2[2] * (2.0 * zz - xx - yy) * sh.index({torch::indexing::Ellipsis, 6}) +
                     C2[3] * xz * sh.index({torch::indexing::Ellipsis, 7}) +
                     C2[4] * (xx - yy) * sh.index({torch::indexing::Ellipsis, 8});
            if (deg > 2) {
                result = result + C3[0] * y * (3 * xx - yy) * sh.index({torch::indexing::Ellipsis, 9}) +
                         C3[1] * xy * z * sh.index({torch::indexing::Ellipsis, 10}) +
                         C3[2] * y * (4 * zz - xx - yy) * sh.index({torch::indexing::Ellipsis, 11}) +
                         C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh.index({torch::indexing::Ellipsis, 12}) +
                         C3[4] * x * (4 * zz - xx - yy) * sh.index({torch::indexing::Ellipsis, 13}) +
                         C3[5] * z * (xx - yy) * sh.index({torch::indexing::Ellipsis, 14}) +
                         C3[6] * x * (xx - 3 * yy) * sh.index({torch::indexing::Ellipsis, 15});
                if (deg > 3) {
                    result = result + C4[0] * xy * (xx - yy) * sh.index({torch::indexing::Ellipsis, 16}) +
                             C4[1] * yz * (3 * xx - yy) * sh.index({torch::indexing::Ellipsis, 17}) +
                             C4[2] * xy * (7 * zz - 1) * sh.index({torch::indexing::Ellipsis, 18}) +
                             C4[3] * yz * (7 * zz - 3) * sh.index({torch::indexing::Ellipsis, 19}) +
                             C4[4] * (zz * (35 * zz - 30) + 3) * sh.index({torch::indexing::Ellipsis, 20}) +
                             C4[5] * xz * (7 * zz - 3) * sh.index({torch::indexing::Ellipsis, 21}) +
                             C4[6] * (xx - yy) * (7 * zz - 1) * sh.index({torch::indexing::Ellipsis, 22}) +
                             C4[7] * xz * (xx - 3 * yy) * sh.index({torch::indexing::Ellipsis, 23}) +
                             C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh.index({torch::indexing::Ellipsis, 24});
                }
            }
        }
    }

    return result;
}

inline torch::Tensor RGB2SH(const torch::Tensor& rgb) {
    return (rgb - 0.5) / C0;
}

inline torch::Tensor SH2RGB(const torch::Tensor& sh) {
    return sh * C0 + 0.5;
}
