// include/core/torch_shapes.hpp
#pragma once
#include <torch/torch.h>

inline void assert_vec(const torch::Tensor& t, int64_t n, const char* name) {
    TORCH_CHECK(t.dim() == 1 && t.size(0) == n,
                name, " must be a ", n, "-vector, got ", t.sizes());
}
inline void assert_mat(const torch::Tensor& t,
                       int64_t r, int64_t c, const char* name) {
    TORCH_CHECK(t.dim() == 2 && t.size(0) == r && t.size(1) == c,
                name, " must be ", r, "×", c, ", got ", t.sizes());
}
