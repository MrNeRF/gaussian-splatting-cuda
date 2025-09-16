/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/logger.hpp"

#include <c10/cuda/CUDAAllocatorConfig.h>
#include <iostream>
#include <print>

int main(int argc, char* argv[]) {
//----------------------------------------------------------------------
// 0. Set CUDA caching allocator settings to avoid fragmentation issues
// This avoids the need to repeatedly call emptyCache() after
// densification steps. We manually call the proper function here
// instead of setting the environment variable hoping that this then
// also works on Windows. Setting the environment variable using
// setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True", 1);
// would work on Linux but not on Windows, so we use the C++ API.
// Should this break in the future, we can always revert to the old
// approach of calling emptyCache() after each densification step.
//----------------------------------------------------------------------
#ifndef _WIN32
    // Windows doesn't support CUDACachingAllocator expandable_segments
    c10::cuda::CUDACachingAllocator::setAllocatorSettings("expandable_segments:True");
#endif

    // Parse arguments (this automatically initializes the logger based on --log-level flag)
    auto params_result = gs::args::parse_args_and_params(argc, argv);
    if (!params_result) {
        // Logger is already initialized, so we can use it for errors
        LOG_ERROR("Failed to parse arguments: {}", params_result.error());
        std::println(stderr, "Error: {}", params_result.error());
        return -1;
    }

    // Logger is now ready to use
    LOG_INFO("========================================");
    LOG_INFO("LichtFeld Studio");
    LOG_INFO("========================================");

    auto params = std::move(*params_result);

    gs::Application app;
    return app.run(std::move(params));
}
