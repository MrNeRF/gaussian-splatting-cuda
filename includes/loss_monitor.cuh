// Copyright (c) 2023 Janusch Patas.
#pragma once
#include <deque>
#include <iostream>

class LossMonitor {

public:
    explicit LossMonitor(size_t size) : _buffer_size(size) {}
    float Update(float newLoss);
    bool IsConverging(float threshold);

private:
    std::deque<float> _loss_buffer;
    std::deque<float> _rate_of_change_buffer;
    size_t _buffer_size;
};
