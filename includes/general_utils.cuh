#pragma once

#include <cmath>
#include <ctime>
#include <iostream>
#include <random>
#include <torch/torch.h>

inline torch::Tensor inverse_sigmoid(torch::Tensor x) {
    return torch::log(x / (1 - x));
}

inline torch::Tensor ImageToTorch(torch::Tensor image, std::vector<int64_t> resolution) {
    torch::Tensor resized_image = image.resize_(resolution) / 255.0;
    if (resized_image.dim() == 3) {
        return resized_image.permute({2, 0, 1});
    } else {
        return resized_image.unsqueeze(-1).permute({2, 0, 1});
    }
}

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

    double operator()(int64_t step) {
        if (step < 0 || (lr_init == 0.0 && lr_final == 0.0)) {
            return 0.0;
        }
        double delay_rate;
        if (lr_delay_steps > 0) {
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * std::sin(0.5 * M_PI * std::min((double)step / (double)lr_delay_steps, 1.0));
        } else {
            delay_rate = 1.0;
        }
        double t = std::min(step / static_cast<double>(max_steps), 1.0);
        double log_lerp = std::exp(std::log(lr_init) * (1 - t) + std::log(lr_final) * t);
        return delay_rate * log_lerp;
    }
};

inline torch::Tensor strip_lowerdiag(torch::Tensor L) {
    torch::Tensor uncertainty = torch::zeros({L.size(0), 6}, torch::dtype(torch::kFloat).device(torch::kCUDA));

    for (int i = 0; i < L.size(0); ++i) {
        uncertainty[i][0] = L[i][0][0];
        uncertainty[i][1] = L[i][0][1];
        uncertainty[i][2] = L[i][0][2];
        uncertainty[i][3] = L[i][1][1];
        uncertainty[i][4] = L[i][1][2];
        uncertainty[i][5] = L[i][2][2];
    }

    return uncertainty;
}

inline torch::Tensor strip_symmetric(torch::Tensor sym) {
    return strip_lowerdiag(sym);
}

inline torch::Tensor build_rotation(torch::Tensor r) {
    torch::Tensor norm = torch::sqrt(torch::sum(r.pow(2), 1));
    torch::Tensor q = r / norm.unsqueeze(-1);

    torch::Tensor R = torch::zeros({q.size(0), 3, 3}, torch::device(torch::kCUDA));

    for (int i = 0; i < q.size(0); ++i) {
        auto r0 = q[i][0], x = q[i][1], y = q[i][2], z = q[i][3];

        R[i][0][0] = 1 - 2 * (y * y + z * z);
        R[i][0][1] = 2 * (x * y - r0 * z);
        R[i][0][2] = 2 * (x * z + r0 * y);
        R[i][1][0] = 2 * (x * y + r0 * z);
        R[i][1][1] = 1 - 2 * (x * x + z * z);
        R[i][1][2] = 2 * (y * z - r0 * x);
        R[i][2][0] = 2 * (x * z - r0 * y);
        R[i][2][1] = 2 * (y * z + r0 * x);
        R[i][2][2] = 1 - 2 * (x * x + y * y);
    }

    return R;
}

inline torch::Tensor build_scaling_rotation(torch::Tensor s, torch::Tensor r) {
    torch::Tensor L = torch::zeros({s.size(0), 3, 3}, torch::dtype(torch::kFloat).device(torch::kCUDA));
    torch::Tensor R = build_rotation(r);

    for (int i = 0; i < s.size(0); ++i) {
        L[i][0][0] = s[i][0];
        L[i][1][1] = s[i][1];
        L[i][2][2] = s[i][2];
    }

    L = torch::matmul(R, L);
    return L;
}
