#pragma once

#include "core/event_bus.hpp"
#include "core/events.hpp"
#include <cuda_runtime.h>
#include <string>

namespace gs {

    class ErrorHandler {
    public:
        explicit ErrorHandler(std::shared_ptr<EventBus> event_bus)
            : event_bus_(event_bus) {}

        // CUDA error checking
        bool checkCudaError(const std::string& operation) {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                publishError(
                    ErrorOccurredEvent::Severity::Error,
                    ErrorOccurredEvent::Category::CUDA,
                    std::string("CUDA error in ") + operation + ": " + cudaGetErrorString(err),
                    std::string("Error code: ") + std::to_string(err),
                    "Try reducing batch size or restarting the application");
                return false;
            }
            return true;
        }

        // General error publishing
        void publishError(
            ErrorOccurredEvent::Severity severity,
            ErrorOccurredEvent::Category category,
            const std::string& message,
            const std::optional<std::string>& details = std::nullopt,
            const std::optional<std::string>& recovery = std::nullopt) {

            if (event_bus_) {
                event_bus_->publish(ErrorOccurredEvent{
                    severity,
                    category,
                    message,
                    details,
                    recovery});
            }
        }

        // Publish recovery
        void publishRecovery(
            ErrorOccurredEvent::Category category,
            const std::string& action) {

            if (event_bus_) {
                event_bus_->publish(ErrorRecoveredEvent{
                    category,
                    action});
            }
        }

    private:
        std::shared_ptr<EventBus> event_bus_;
    };

} // namespace gs
