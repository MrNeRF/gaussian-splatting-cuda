#pragma once

#include <cmath>
#include <ctime>
#include <iostream>
#include <random>
#include <torch/torch.h>

/**
 * Calculates the inverse sigmoid of a tensor element-wise.
 *
 * @param x The input tensor.
 * @return The tensor with the inverse sigmoid of each element.
 */
inline torch::Tensor inverse_sigmoid(torch::Tensor x) {
    return torch::log(x / (1 - x));
}

/**
 * Converts an image tensor to a torch tensor with the given resolution.
 *
 * @param image The input image tensor.
 * @param resolution The desired resolution of the output tensor.
 * @return A torch tensor with the given resolution.
 */
inline torch::Tensor ImageToTorch(torch::Tensor image, std::vector<int64_t> resolution) {
    torch::Tensor resized_image = image.resize_(resolution) / 255.0;
    if (resized_image.dim() == 3) {
        return resized_image.permute({2, 0, 1});
    } else {
        return resized_image.unsqueeze(-1).permute({2, 0, 1});
    }
}

/**
 * @brief A functor that implements an exponential learning rate decay function.
 *
 * This functor is used to implement an exponential learning rate decay function. It takes in the initial learning rate,
 * the final learning rate, the number of steps to delay the decay, the delay multiplier, and the maximum number of steps.
 *
 * @param lr_init The initial learning rate.
 * @param lr_final The final learning rate.
 * @param lr_delay_steps The number of steps to delay the decay.
 * @param lr_delay_mult The delay multiplier.
 * @param max_steps The maximum number of steps.
 *
 * @return The learning rate at the given step.
 */
struct Expon_lr_func {
    double lr_init;
    double lr_final;
    int64_t lr_delay_steps;
    double lr_delay_mult;
    int64_t max_steps;
    Expon_lr_func(double lr_init, double lr_final, int64_t lr_delay_steps = 0, double lr_delay_mult = 1.0, int64_t max_steps = 1000000)
        : lr_init(lr_init),
          lr_final(lr_final),
          lr_delay_steps(lr_delay_steps),
          lr_delay_mult(lr_delay_mult),
          max_steps(max_steps) {}

    double operator()(int64_t step) const {
        if (step < 0 || (lr_init == 0.0 && lr_final == 0.0)) {
            return 0.0;
        }
        double delay_rate;
        if (lr_delay_steps > 0 && step != 0) {
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * std::sin(0.5 * M_PI * std::min((double)step / (double)lr_delay_steps, 1.0));
        } else {
            delay_rate = 1.0;
        }
        double t = std::min(step / static_cast<double>(max_steps), 1.0);
        double log_lerp = std::exp(std::log(lr_init) * (1 - t) + std::log(lr_final) * t);
        return delay_rate * log_lerp;
    }
};

/**
 * @brief Strips the lower diagonal elements of a 3x3 matrix and returns them as a 1D tensor of size (N, 6).
 *
 * @param L A 3D tensor of size (N, 3, 3) representing a batch of 3x3 matrices.
 * @return A 2D tensor of size (N, 6) representing the lower diagonal elements of each matrix in L.
 */
inline torch::Tensor strip_lowerdiag(torch::Tensor L) {
    torch::Tensor uncertainty = torch::zeros({L.size(0), 6}, torch::dtype(torch::kFloat).device(torch::kCUDA));
    using Slice = torch::indexing::Slice;

    uncertainty.index_put_({Slice(), 0}, L.index({Slice(), 0, 0}));
    uncertainty.index_put_({Slice(), 1}, L.index({Slice(), 0, 1}));
    uncertainty.index_put_({Slice(), 2}, L.index({Slice(), 0, 2}));
    uncertainty.index_put_({Slice(), 3}, L.index({Slice(), 1, 1}));
    uncertainty.index_put_({Slice(), 4}, L.index({Slice(), 1, 2}));
    uncertainty.index_put_({Slice(), 5}, L.index({Slice(), 2, 2}));
    return uncertainty;
}

/**
 * @brief Strips the symmetric part of a tensor.
 *
 * This function takes an input tensor and returns the tensor with the symmetric part removed.
 *
 * @param sym The input tensor.
 * @return The tensor with the symmetric part removed.
 */
inline torch::Tensor strip_symmetric(torch::Tensor sym) {
    return strip_lowerdiag(sym);
}

/**
 * @brief Builds a rotation matrix from a tensor of quaternions.
 *
 * @param r Tensor of quaternions with shape (N, 4).
 * @return Tensor of rotation matrices with shape (N, 3, 3).
 */
inline torch::Tensor build_rotation(torch::Tensor r) {
    torch::Tensor norm = torch::sqrt(torch::sum(r.pow(2), 1));
    torch::Tensor q = r / norm.unsqueeze(-1);

    using Slice = torch::indexing::Slice;
    torch::Tensor R = torch::zeros({q.size(0), 3, 3}, torch::device(torch::kCUDA));
    torch::Tensor r0 = q.index({Slice(), 0});
    torch::Tensor x = q.index({Slice(), 1});
    torch::Tensor y = q.index({Slice(), 2});
    torch::Tensor z = q.index({Slice(), 3});

    R.index_put_({Slice(), 0, 0}, 1 - 2 * (y * y + z * z));
    R.index_put_({Slice(), 0, 1}, 2 * (x * y - r0 * z));
    R.index_put_({Slice(), 0, 2}, 2 * (x * z + r0 * y));
    R.index_put_({Slice(), 1, 0}, 2 * (x * y + r0 * z));
    R.index_put_({Slice(), 1, 1}, 1 - 2 * (x * x + z * z));
    R.index_put_({Slice(), 1, 2}, 2 * (y * z - r0 * x));
    R.index_put_({Slice(), 2, 0}, 2 * (x * z - r0 * y));
    R.index_put_({Slice(), 2, 1}, 2 * (y * z + r0 * x));
    R.index_put_({Slice(), 2, 2}, 1 - 2 * (x * x + y * y));
    return R;
}

/**
 * Builds a scaling-rotation matrix from the given scaling and rotation tensors.
 *
 * @param s The scaling tensor of shape (N, 3) where N is the number of scaling factors.
 * @param r The rotation tensor of shape (N, 3) where N is the number of rotation angles.
 * @return The scaling-rotation matrix of shape (N, 3, 3) where N is the number of scaling-rotation matrices.
 */
inline torch::Tensor build_scaling_rotation(torch::Tensor s, torch::Tensor r) {
    torch::Tensor L = torch::zeros({s.size(0), 3, 3}, torch::dtype(torch::kFloat).device(torch::kCUDA));
    torch::Tensor R = build_rotation(r);

    using Slice = torch::indexing::Slice;
    L.index_put_({Slice(), 0, 0}, s.index({Slice(), 0}));
    L.index_put_({Slice(), 1, 1}, s.index({Slice(), 1}));
    L.index_put_({Slice(), 2, 2}, s.index({Slice(), 2}));

    L = R.matmul(L);
    return L;
}
