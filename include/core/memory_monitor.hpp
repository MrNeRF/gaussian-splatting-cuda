#pragma once

#include "core/event_bus.hpp"
#include "core/events.hpp"
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <memory>
#include <thread>

namespace gs {

    class MemoryMonitor {
    public:
        explicit MemoryMonitor(std::shared_ptr<EventBus> event_bus)
            : event_bus_(event_bus) {}

        ~MemoryMonitor() {
            stop();
        }

        void start(std::chrono::milliseconds interval = std::chrono::milliseconds(1000)) {
            if (running_)
                return;

            running_ = true;
            monitor_thread_ = std::thread([this, interval]() {
                while (running_) {
                    updateMemoryStats();
                    std::this_thread::sleep_for(interval);
                }
            });
        }

        void stop() {
            if (running_) {
                running_ = false;
                if (monitor_thread_.joinable()) {
                    monitor_thread_.join();
                }
            }
        }

        void updateMemoryStats() {
            // GPU Memory
            size_t gpu_free, gpu_total;
            cudaMemGetInfo(&gpu_free, &gpu_total);
            size_t gpu_used = gpu_total - gpu_free;
            float gpu_percent = (float)gpu_used / gpu_total * 100.0f;

            // RAM (simplified - platform specific implementation needed)
            size_t ram_used = 0, ram_total = 0;
            float ram_percent = 0.0f;
            // TODO: Implement platform-specific RAM monitoring

            // Publish memory usage event
            if (event_bus_) {
                event_bus_->publish(MemoryUsageEvent{
                    gpu_used,
                    gpu_total,
                    gpu_percent,
                    ram_used,
                    ram_total,
                    ram_percent});

                // Check for warnings
                if (gpu_percent > 90.0f && !gpu_warning_sent_) {
                    event_bus_->publish(MemoryWarningEvent{
                        MemoryWarningEvent::Type::GPU,
                        gpu_percent,
                        "GPU memory usage critical! Consider reducing batch size or number of Gaussians."});
                    gpu_warning_sent_ = true;
                } else if (gpu_percent < 85.0f) {
                    gpu_warning_sent_ = false;
                }
            }
        }

    private:
        std::shared_ptr<EventBus> event_bus_;
        std::thread monitor_thread_;
        std::atomic<bool> running_{false};
        bool gpu_warning_sent_ = false;
    };

} // namespace gs
