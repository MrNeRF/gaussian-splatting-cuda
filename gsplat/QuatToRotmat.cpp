#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"       // where all the macros are defined
#include "Ops.h"          // a collection of all gsplat operators
#include "QuatToRotmat.h" // where the launch function is declared

namespace gsplat {

    at::Tensor quats_to_rotmats(
        const at::Tensor quats // [N, 4]
    ) {
        DEVICE_GUARD(quats);
        CHECK_INPUT(quats);

        uint32_t N = quats.size(0);

        at::Tensor rotmats = at::empty({N, 3, 3}, quats.options());
        launch_quats_to_rotmats_kernel(quats, rotmats);

        return rotmats;
    }

} // namespace gsplat
