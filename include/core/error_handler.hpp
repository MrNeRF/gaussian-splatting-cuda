#pragma once

#include "core/events.hpp"
#include <cuda_runtime.h>
#include <string>

namespace gs {

    class ErrorHandler {
    public:
        ErrorHandler() = default;

        // CUDA error checking
        bool checkCudaError(const std::string& operation) {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                events::notify::Error{
                    .message = std::string("CUDA error in ") + operation + ": " + cudaGetErrorString(err),
                    .details = std::string("Error code: ") + std::to_string(err)}
                    .emit();
                return false;
            }
            return true;
        }

        // General error publishing
        void publishError(const std::string& message, const std::string& details = "") {
            events::notify::Error{
                .message = message,
                .details = details}
                .emit();
        }
    };

} // namespace gs