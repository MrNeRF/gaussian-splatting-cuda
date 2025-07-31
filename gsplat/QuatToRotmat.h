#pragma once

#include <cstdint>

namespace at {
    class Tensor;
}

namespace gsplat {

    void launch_quats_to_rotmats_kernel(
        // inputs
        const at::Tensor quats, // [N, 4]
        // outputs
        at::Tensor rotmat // [N, 3, 3]
    );

} // namespace gsplat
