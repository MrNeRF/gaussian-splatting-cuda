/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/logger.hpp"

#include <c10/cuda/CUDAAllocatorConfig.h>
#include <cuda_runtime.h>
#include <iostream>
#include <print>

int main(int argc, char* argv[]) {
    //----------------------------------------------------------------------
    // Configure CUDA memory allocator for optimal performance
    //----------------------------------------------------------------------

#ifdef _WIN32
    // Windows-specific optimizations since expandable_segments isn't available
    // These settings help reduce fragmentation:
    std::string allocator_config =
        "max_split_size_mb:256,"            // Smaller max split size to reduce fragmentation
        "garbage_collection_threshold:0.5," // More aggressive garbage collection (default is 0.8)
        "roundup_power2_divisions:4";       // Better size class matching

    try {
        c10::cuda::CUDACachingAllocator::setAllocatorSettings(allocator_config);
        std::println("Applied Windows-optimized CUDA allocator settings");
    } catch (const std::exception& e) {
        std::println(stderr, "Warning: Could not set allocator settings: {}", e.what());
    }
#else
    // Linux/Unix with expandable segments support
    try {
        // Primary setting for Linux - prevents fragmentation
        c10::cuda::CUDACachingAllocator::setAllocatorSettings("expandable_segments:True");
        std::println("Enabled expandable segments for CUDA allocator");
    } catch (const std::exception& e) {
        // Fallback if expandable segments not supported (older CUDA)
        std::string fallback_config =
            "max_split_size_mb:512,"
            "garbage_collection_threshold:0.6";
        try {
            c10::cuda::CUDACachingAllocator::setAllocatorSettings(fallback_config);
            std::println("Applied fallback CUDA allocator settings");
        } catch (...) {
            std::println(stderr, "Warning: Could not configure CUDA allocator");
        }
    }
#endif

    // Parse arguments (this automatically initializes the logger based on --log-level flag)
    auto params_result = gs::args::parse_args_and_params(argc, argv);
    if (!params_result) {
        LOG_ERROR("Failed to parse arguments: {}", params_result.error());
        std::println(stderr, "Error: {}", params_result.error());
        return -1;
    }

    // Logger is now ready to use
    LOG_INFO("========================================");
    LOG_INFO("LichtFeld Studio");
    LOG_INFO("========================================");

    // Log memory configuration
#ifdef _WIN32
    LOG_INFO("Memory: Windows mode (periodic cache clearing enabled)");
#else
    LOG_INFO("Memory: Linux mode (expandable segments enabled)");
#endif

    auto params = std::move(*params_result);

    // Additional memory debugging info if requested
    if (params->optimization.debug_memory) {
        size_t free, total;
        cudaError_t err = cudaMemGetInfo(&free, &total);
        if (err == cudaSuccess) {
            LOG_INFO("Initial GPU memory: {:.2f}GB free of {:.2f}GB total",
                     free / (1024.0 * 1024.0 * 1024.0),
                     total / (1024.0 * 1024.0 * 1024.0));
        }
    }

    gs::Application app;
    return app.run(std::move(params));
}
