#pragma once
#include "core/cam_debug.hpp"     // uses dump_tensor / dump_scalar
#include "rasterizer.hpp"         // for GaussianRasterizationSettings
#include <string>

namespace renderdbg {

    inline void dump_settings(const GaussianRasterizationSettings& s,
                              const std::string& tag,
                              const std::string& id)        // e.g. cam-uid
    {
        using camdbg::dump_tensor;
        using camdbg::dump_scalar;

        // scalars ----------------------------------------------------------
        dump_scalar(float(s.image_height), "H_" + id, tag);
        dump_scalar(float(s.image_width ), "W_" + id, tag);
        dump_scalar(s.tanfovx,             "tfx_" + id, tag);
        dump_scalar(s.tanfovy,             "tfy_" + id, tag);
        dump_scalar(s.scale_modifier,      "scale_" + id, tag);
        dump_scalar(float(s.sh_degree),    "shdeg_" + id, tag);
        dump_scalar(float(s.prefiltered),  "pref_" + id, tag); // 0/1

        // tensors ----------------------------------------------------------
        dump_tensor(s.bg.cpu(),         "bg_"  + id, tag);      // 3
        dump_tensor(s.viewmatrix.cpu(), "wv_"  + id, tag);      // 4×4
        dump_tensor(s.projmatrix.cpu(), "p_"   + id, tag);      // 4×4
        dump_tensor(s.camera_center.cpu(),"cc_" + id, tag);     // 3
    }

} // namespace renderdbg
