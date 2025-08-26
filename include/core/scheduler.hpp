#pragma once

#include <torch/torch.h>

// Simple ExponentialLR implementation since C++ API is different
class ExponentialLR {
public:
    ExponentialLR(torch::optim::Optimizer& optimizer, double gamma, int param_group_index = -1)
        : optimizer_(optimizer),
          gamma_(gamma),
          param_group_index_(param_group_index) {}

    void step();

private:
    torch::optim::Optimizer& optimizer_;
    double gamma_;
    int param_group_index_;
};