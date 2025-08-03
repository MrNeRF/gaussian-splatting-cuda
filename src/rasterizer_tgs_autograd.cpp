#include "Ops.h"
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include <torch/torch.h>
#include "core/rasterizer_tgs_autograd.hpp"
#include "rasterize_points.h" 
#include <cmath>

namespace tgs {

    torch::autograd::variable_list RasterizationFunction::forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& means3D,
        const torch::Tensor& sh0,
        const torch::Tensor& shN,
        const torch::Tensor& colors_precomp,
        const torch::Tensor& opacities,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& cov3Ds_precomp,
        const torch::Tensor& viewmat,      // [B,4,4]
        const torch::Tensor& projmat,           // [B,3,3]
        const torch::Tensor& bg_color,     // [B,3] or [3]
        const torch::Tensor& campos,
        const torch::Tensor& settings)     // [5]  {scale_mod, H, W, sh_degree, antialias}
    {

        ctx->set_materialize_grads(false);

        const float scaling_modifier = settings[0].item<float>();
        const int   image_height     = settings[1].item<int>();
        const int   image_width      = settings[2].item<int>();
        const int   sh_degree        = settings[3].item<int>();
        const bool  antialiased      = settings[4].item<bool>();
        const float tan_fovx         = settings[5].item<float>(); 
        const float tan_fovy         = settings[6].item<float>();

        auto [
            num_rendered,
            num_buckets,
            color,
            radii,
            geomBuf,
            binningBuf,
            imgBuf,
            sampleBuf,
            offsetBuf,
            listBuf,
            listBufR,
            listBufD,
            xy_d,
            depths_d,
            radii_d,
            acc_w,
            acc_c,
            acc_b,
            acc_d
        ] = taminggs::RasterizeGaussiansCUDA(
            /*background   */ bg_color,
            /*means        */ means3D,
            /*colors       */ colors_precomp,
            /*opacity      */ opacities,
            /*scales       */ scales,
            /*rotations    */ rotations,
            /*scale_mod    */ scaling_modifier,
            /*cov3D        */ cov3Ds_precomp,
            /*viewmatrix   */ viewmat.transpose(1, 2), // glm expects column major
            /*projmatrix   */ projmat.transpose(1, 2), // glm expects column major
            /*tan_fovx     */ tan_fovx,
            /*tan_fovy     */ tan_fovy,
            /*H, W         */ image_height, image_width,
            /*dc           */ sh0,
            /*shs          */ shN,
            /*degree       */ sh_degree,
            /*campos       */ campos,
            /*prefiltered  */ antialiased,
            /*debug        */ false,
            /*pixel_weights*/ torch::empty({0}, torch::kFloat32)
        );

        auto grad_settings = torch::tensor({
            scaling_modifier,
            (float)sh_degree,
            (float)antialiased,
            tan_fovx, tan_fovy,
            (float)num_rendered, (float)num_buckets },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
        );

        ctx->save_for_backward({
            means3D, scales, rotations, sh0, shN,
            viewmat, projmat, campos, bg_color,
            colors_precomp, cov3Ds_precomp, radii,
            geomBuf, imgBuf, binningBuf, sampleBuf,
            grad_settings
        });

        // image, depth, radii
        return {color, radii, geomBuf, binningBuf, imgBuf, sampleBuf,
                offsetBuf, listBuf, listBufR, listBufD, xy_d, depths_d,
                radii_d, acc_w, acc_c, acc_b, acc_d};

        }

    torch::autograd::variable_list RasterizationFunction::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs)
    {

        torch::Tensor grad_out_color = grad_outputs[0];

        // 1) Grab the vector of saved tensors
        auto saved = ctx->get_saved_variables();

        // 2) Unpack in the same order you saved them
        auto means3D        = saved[0];
        auto scales         = saved[1];
        auto rotations      = saved[2];
        auto sh0             = saved[3];
        auto shN            = saved[4];
        auto viewmat        = saved[5];
        auto projmat        = saved[6];
        auto campos         = saved[7];
        auto bg_color       = saved[8];
        auto colors_precomp = saved[9];
        auto cov3Ds_precomp = saved[10];
        auto radii          = saved[11];
        auto geomBuf        = saved[12];
        auto imgBuf         = saved[13];
        auto binningBuf     = saved[14];
        auto sampleBuf      = saved[15];

        // finally, your grad_settings tensor is at index 12
        auto grad_settings  = saved[16];

        // 3) Pull individual floats/ints/bools back out of grad_settings
        float scaling_modifier = grad_settings[0].item<float>();
        int   sh_degree        = static_cast<int>(grad_settings[1].item<float>());
        bool  antialiased      = static_cast<bool>(grad_settings[2].item<float>());
        float tan_fovx         = grad_settings[3].item<float>();
        float tan_fovy         = grad_settings[4].item<float>();
        int   num_rendered     = static_cast<int>(grad_settings[5].item<float>());
        int   num_buckets      = static_cast<int>(grad_settings[6].item<float>());

        // 4) Now call your CUDA backward kernel, re‑using all of those
        auto [
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
            grad_cov3Ds_precomp, grad_sh0, grad_shN, grad_scales, grad_rotations
        ] = taminggs::RasterizeGaussiansBackwardCUDA(
            /*bg*/        bg_color,
            /*means3D*/   means3D,
            /*radii*/     radii,
            /*colors*/    colors_precomp,
            /*scales*/    scales,
            /*rots*/      rotations,
            /*scale_mod*/ scaling_modifier,
            /*cov3D*/     cov3Ds_precomp,
            /*viewmat*/   viewmat.transpose(1,2),
            /*projmat*/   projmat.transpose(1,2),
            /*tan x*/     tan_fovx,
            /*tan y*/     tan_fovy,
            /*dL/dout*/   grad_out_color,
            /*dc*/        sh0,
            /*sh*/        shN,
            /*degree*/    sh_degree,
            /*campos*/    campos,
            /*geomBuf*/   geomBuf,
            /*R*/         num_rendered,
            /*binBuf*/    binningBuf,
            /*imgBuf*/    imgBuf,
            /*B*/         num_buckets,
            /*sampBuf*/   sampleBuf,
            /*debug*/     false
        );

        return {
            grad_means3D,
            grad_sh0,
            grad_shN,
            grad_colors_precomp,
            grad_opacities.squeeze(1),
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()
        };

    }

}