// Copyright (c) 2023 Janusch Patas.

#include "loss_monitor.cuh"
#include <numeric>

float LossMonitor::Update(float newLoss) {
    if (_loss_buffer.size() >= _buffer_size) {
        _loss_buffer.pop_front();
        _rate_of_change_buffer.pop_front();
    }
    const bool buffer_empty = _loss_buffer.empty();
    const float rateOfChange = buffer_empty ? 0.f : std::abs(newLoss - _loss_buffer.back());
    _rate_of_change_buffer.push_back(rateOfChange);
    _loss_buffer.push_back(newLoss);

    // return average rate of change
    return buffer_empty ? 0.f : std::accumulate(_rate_of_change_buffer.begin(), _rate_of_change_buffer.end(), 0.f) / _rate_of_change_buffer.size();
}

bool LossMonitor::IsConverging(float threshold) {
    if (_rate_of_change_buffer.size() < _buffer_size) {
        return false;
    }
    return std::accumulate(_rate_of_change_buffer.begin(), _rate_of_change_buffer.end(), 0.f) / _rate_of_change_buffer.size() <= threshold;
}
