// Copyright (c) 2023 Janusch Patas.

#include "loss_monitor.cuh"
#include <numeric>

float LossMonitor::Update(float newLoss) {
    if (_loss_buffer.size() >= _buffer_size) {
        _loss_buffer.pop_front();
        _rate_of_change_buffer.pop_front();
    }
    float rateOfChange = (_loss_buffer.empty()) ? 0.0 : std::abs(newLoss - _loss_buffer.back());
    _rate_of_change_buffer.push_back(rateOfChange);
    _loss_buffer.push_back(newLoss);
    return std::accumulate(_rate_of_change_buffer.begin(), _rate_of_change_buffer.end(), 0.0) / _rate_of_change_buffer.size();
}
bool LossMonitor::IsConverging(float threshold) {
    return Update(_loss_buffer.back()) < threshold;
}
