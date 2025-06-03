#pragma once
// -----------------------------------------------------------------------------
// Header-only LibTorch wrapper for *Fused SSIM*
//  – The prototype is visible to both host & device compilations.
//  – The implementation is compiled **only** on the host side
//    (i.e. when __CUDA_ARCH__ is *not* defined or equals 0).
// -----------------------------------------------------------------------------
#include "kernels/ssim.cuh" // declares fusedssim & fusedssim_backward
#include <torch/torch.h>
#include <tuple>

namespace fs_internal {
    constexpr double kC1 = 0.01 * 0.01;
    constexpr double kC2 = 0.03 * 0.03;

    inline void check_padding(const std::string& p) {
        TORCH_CHECK(p == "same" || p == "valid",
                    "fused_ssim: padding must be \"same\" or \"valid\" (got \"",
                    p, "\")");
    }

    /* ----------------- custom autograd Function (always header-only) --------- */
    class _FusedSSIM : public torch::autograd::Function<_FusedSSIM> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                     torch::Tensor img1,
                                     torch::Tensor img2,
                                     const std::string& padding,
                                     bool train) {
            check_padding(padding);
            img1 = img1.contiguous();
            img2 = img2.contiguous();

            // Ensure 4D tensors [N, C, H, W]
            if (img1.dim() == 3) {
                img1 = img1.unsqueeze(0); // Add batch dimension
            }
            if (img2.dim() == 3) {
                img2 = img2.unsqueeze(0); // Add batch dimension
            }

            // Verify dimensions
            TORCH_CHECK(img1.dim() == 4 && img2.dim() == 4,
                        "fused_ssim expects 4D tensors [N,C,H,W], got img1: ",
                        img1.dim(), "D, img2: ", img2.dim(), "D");
            TORCH_CHECK(img1.sizes() == img2.sizes(),
                        "img1 and img2 must have the same shape");

            auto out = fusedssim(kC1, kC2, img1, img2, train);
            auto map = std::get<0>(out);
            auto dm1 = std::get<1>(out);
            auto ds1sq = std::get<2>(out);
            auto ds12 = std::get<3>(out);

            if (padding == "valid") {
                using torch::indexing::Slice;
                // Convert negative indices to positive ones
                int64_t h = map.size(2); // height
                int64_t w = map.size(3); // width
                if (h > 10 && w > 10) {  // Ensure we have enough pixels to crop
                    map = map.index({Slice(), Slice(), Slice(5, h - 5), Slice(5, w - 5)});
                }
            }

            ctx->save_for_backward({img1.detach(), img2, dm1, ds1sq, ds12});
            ctx->saved_data["padding"] = padding;
            return map;
        }

        static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext* ctx,
                                                   std::vector<torch::Tensor> grad_out) {
            auto vars = ctx->get_saved_variables();
            auto img1 = vars[0];
            auto img2 = vars[1];
            auto dm1 = vars[2];
            auto ds1sq = vars[3];
            auto ds12 = vars[4];
            std::string padding = ctx->saved_data["padding"].toStringRef();

            auto dL_dmap = grad_out[0];
            if (padding == "valid") {
                using torch::indexing::Slice;
                auto full = torch::zeros_like(img1);
                // Convert negative indices to positive ones
                int64_t h = full.size(2); // height
                int64_t w = full.size(3); // width
                if (h > 10 && w > 10) {   // Ensure we have enough pixels to crop
                    full.index_put_({Slice(), Slice(), Slice(5, h - 5), Slice(5, w - 5)},
                                    dL_dmap);
                }
                dL_dmap = full;
            }

            auto grad_img1 = fusedssim_backward(
                kC1, kC2, img1, img2, dL_dmap, dm1, ds1sq, ds12);

            return {grad_img1, torch::Tensor(), torch::Tensor(), torch::Tensor()};
        }
    };
} // namespace fs_internal

// ---------------------------------------------------------------------------
// ALWAYS-VISIBLE PROTOTYPE  ➜ lets both host & device compilers see the symbol
// ---------------------------------------------------------------------------
torch::Tensor fused_ssim(torch::Tensor img1, torch::Tensor img2,
                         const std::string& padding = "same",
                         bool train = true);

// ---------------------------------------------------------------------------
// HOST-ONLY IMPLEMENTATION  ➜ excluded from device compilation
// ---------------------------------------------------------------------------
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0)
inline torch::Tensor fused_ssim(torch::Tensor img1, torch::Tensor img2,
                                const std::string& padding, bool train) {
    fs_internal::check_padding(padding);
    img1 = img1.contiguous();
    return fs_internal::_FusedSSIM::apply(img1, img2, padding, train).mean();
}
#endif // !__CUDA_ARCH__